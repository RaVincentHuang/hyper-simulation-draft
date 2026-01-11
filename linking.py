import itertools
import re
from embedding import get_similarity_batch
from nli import get_nli_entailment_score_batch
import requests
import concurrent.futures
from collections import defaultdict
import conceptnet_lite

# # 这行代码建议放在程序的入口处，只需执行一次
conceptnet_lite.connect("conceptnet.db")
# ==========================================
# 模拟相似度接口 (实际请接入你的 BERT/Embedding 模型)
# ==========================================
class WikidataTagger:
    def __init__(self, max_workers=10):
        self.headers = {'User-Agent': 'Bot/1.0 (Contact: your_email@example.com)'}
        self.wd_api = "https://www.wikidata.org/w/api.php"
        self.max_workers = max_workers

        # =========================================================
        # 配置：Wikidata 属性映射
        # 只保留 Wikidata 的 P-Code
        # =========================================================
        self.LABEL_MAP = {
            'P31':   'WD:InstanceOf',       # 是...的实例
            'P279':  'WD:SubclassOf',       # 是...的子类
            'P366':  'WD:HasUse',           # 用途
            'P101':  'WD:FieldOfWork',     # 所属领域
            'P361':  'WD:PartOf',           # 是...的一部分
            'P527':  'WD:HasPart',          # 包含...
            'P131':  'WD:LocatedIn',        # 位于
            'P495':  'WD:CountryOfOrigin', # 原产地
            'P1552': 'WD:HasQuality',       # 具有特征 (Has quality)
            'P1056': 'WD:Characteristic'     # 产品特性 (Product characteristic)
        }

    def batch_process(self, pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
        """
        处理一批 (Term, Context) 对。
        返回列表长度与输入一致，每个结果独立消歧。
        """
        # 初始化结果容器
        temp_results = [defaultdict(list) for _ in range(len(pairs))]
        
        # 提取 Unique Term 减少搜索请求
        unique_terms = list(set(t for t, s in pairs))
        
        # ---------------------------------------------------------
        # Phase 1: 并发获取 Wikidata 候选 (Candidate Search)
        # ---------------------------------------------------------
        print(f"Phase 1: Searching candidates for {len(unique_terms)} unique terms...")
        wd_candidates_map = {} 
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_wd = {executor.submit(self._search_candidates, t): t for t in unique_terms}
            for future in concurrent.futures.as_completed(future_wd):
                wd_candidates_map[future_wd[future]] = future.result()

        # ---------------------------------------------------------
        # Phase 2: 批量消歧 (Batch Disambiguation)
        # ---------------------------------------------------------
        print("Phase 2: Running batch disambiguation...")
        
        batch_queries = []
        batch_data = []
        batch_mapping = [] # (index, candidate_object)

        for i, (term, context) in enumerate(pairs):
            cands = wd_candidates_map.get(term, [])
            for cand in cands:
                # Query: 上下文
                # Data: 实体的 Label + Description
                batch_queries.append(context)
                batch_data.append(f"{cand['label']} {cand['desc']}")
                batch_mapping.append((i, cand))
        
        # 如果没有候选，直接返回空结果
        if not batch_queries:
            return self._format_output(temp_results)
            
        scores = get_nli_entailment_score_batch(list(zip(batch_queries, batch_data)))
                
        # 解析分数：找到每个 index 对应的最佳 QID
        index_best_match = {} # { input_index: (max_score, qid) }
        
        for score, (original_index, cand) in zip(scores, batch_mapping):
            print(f"Debug: Index {original_index}, Candidate '{cand['label']}', Score: {score:.4f}")
            if original_index not in index_best_match or score > index_best_match[original_index][0]:
                index_best_match[original_index] = (score, cand['id'])

        # ---------------------------------------------------------
        # Phase 3: 获取详细属性 (Deep Fetch)
        # ---------------------------------------------------------
        print("Phase 3: Fetching deep details...")
        
        # 收集需要查询的 Unique QID
        needed_qids = list(set(qid for _, qid in index_best_match.values()))
        qid_details_map = {} 

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_qid = {executor.submit(self._fetch_details, qid): qid for qid in needed_qids}
            for future in concurrent.futures.as_completed(future_to_qid):
                qid = future_to_qid[future]
                qid_details_map[qid] = future.result()

        # 回填数据到对应的 input index
        for i in range(len(pairs)):
            if i in index_best_match:
                best_qid = index_best_match[i][1]
                details = qid_details_map.get(best_qid, {})
                for k, v in details.items():
                    temp_results[i][k].extend(v)

        return self._format_output(temp_results)

    def _format_output(self, temp_results):
        """格式化输出：去重并用分号连接"""
        final_output = []
        for res in temp_results:
            flat_dict = {}
            for k, v_list in res.items():
                unique_vals = sorted(list(set(v_list)))
                if unique_vals:
                    flat_dict[k] = "; ".join(unique_vals)
            final_output.append(flat_dict)
        return final_output

    # ==========================================
    # 内部 Wikidata IO 方法
    # ==========================================
    def _search_candidates(self, term):
        """根据词语搜索 Wikidata 实体候选"""
        params = {"action": "wbsearchentities", "format": "json", "language": "en", "search": term, "limit": 5}
        try:
            res = requests.get(self.wd_api, params=params, headers=self.headers).json()
            return [{"id": x["id"], "label": x.get("label", ""), "desc": x.get("description", "")} for x in res.get("search", [])]
        except: return []

    def _fetch_details(self, qid):
        """根据 QID 获取详细的 Claims，并映射为 Label"""
        result = defaultdict(list)
        params = {"action": "wbgetentities", "ids": qid, "format": "json", "languages": "en", "props": "claims"}
        try:
            data = requests.get(self.wd_api, params=params, headers=self.headers).json()
            claims = data.get('entities', {}).get(qid, {}).get('claims', {})
            
            target_ids = []
            id_context = [] # (val_qid, output_key)
            
            # 遍历配置表
            for pid, output_key in self.LABEL_MAP.items():
                if pid in claims:
                    for stmt in claims[pid]:
                        # 确保是实体链接 (wikibase-item)
                        if stmt.get('mainsnak', {}).get('datatype') == 'wikibase-item':
                            try:
                                val_qid = stmt['mainsnak']['datavalue']['value']['id']
                                target_ids.append(val_qid)
                                id_context.append((val_qid, output_key))
                            except: continue
            
            # 批量转 Label
            if target_ids:
                label_map = self._ids_to_labels(target_ids)
                for t_qid, key in id_context:
                    if t_qid in label_map:
                        result[key].append(label_map[t_qid])
        except: pass
        return result

    def _ids_to_labels(self, qids):
        """批量将 QID 转为英文 Label"""
        unique_ids = list(set(qids))
        label_map = {}
        # 分块处理，每块 50 个
        for i in range(0, len(unique_ids), 50):
            chunk = unique_ids[i:i+50]
            params = {"action": "wbgetentities", "ids": "|".join(chunk), "format": "json", "props": "labels", "languages": "en"}
            try:
                res = requests.get(self.wd_api, params=params, headers=self.headers).json()
                for qid, ent in res.get('entities', {}).items():
                    lbl = ent.get('labels', {}).get('en', {}).get('value')
                    if lbl: label_map[qid] = lbl
            except: pass
        return label_map

class ConceptNetTagger:
    def __init__(self, db_path="conceptnet.db"):
        print(f"Connecting to local ConceptNet DB: {db_path}...")
        try:
            conceptnet_lite.connect(db_path)
        except Exception as e:
            print(f"Connection warning: {e}")

    def batch_process(self, pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
        final_results = [{"type": None} for _ in range(len(pairs))]
        unique_terms = list(set(t for t, s in pairs))
        
        print(f"Phase 1: Fetching candidates for {len(unique_terms)} terms...")
        candidates_map = {} 
        for term in unique_terms:
            candidates_map[term] = self._fetch_isa_candidates(term)

        print("Phase 2: Constructing and Ranking...")
        batch_queries = []
        batch_data = []
        # meta 存更多信息: (index, candidate_label, is_single_word)
        batch_meta = [] 

        for i, (term, context) in enumerate(pairs):
            cands = candidates_map.get(term, set())
            
            if not cands: 
                print(f"  Warning: No candidates for '{term}'.")
                continue

            for cand in cands:
                # 标记是否为单个词
                is_single = len(cand.split()) == 1
                
                # 构造 Prompt
                batch_queries.append(context)
                hypothesis = f"{term} is a {cand}."
                batch_data.append(hypothesis)
                batch_meta.append((i, cand, is_single))
        
        if not batch_queries:
            return final_results
            
        scores = get_nli_entailment_score_batch(list(zip(batch_queries, batch_data)))
        
        # ---------------------------------------------------------------
        # Phase 3: 分级择优策略 (Tiered Selection)
        # ---------------------------------------------------------------
        # 1. 把所有结果按 input index 分组
        grouped_results = defaultdict(list)
        for score, (idx, label, is_single) in zip(scores, batch_meta):
            if score > 0.1: # 基础过滤阈值
                grouped_results[idx].append({
                    "label": label,
                    "score": score,
                    "is_single": is_single
                })
        
        print("\n--- Ranking Details ---")
        for idx, candidates in grouped_results.items():
            original_term = pairs[idx][0]
            
            # 2. 按分数从高到低排序
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            for c in candidates:
                print(f"[{original_term}] Candidate: '{c['label']}' (Score: {c['score']}, Single: {c['is_single']})")
            
            if not candidates: continue
            
            # 获取最高分 (作为一个基准)
            top_score = candidates[0]['score']
            
            selected_cand = None
            
            # 3. 优先策略：
            # 在分数接近最高分 (例如 top_score - 0.05 范围内) 的候选词中，
            # 优先找 Single Word。如果找不到，才接受 Multi-word。
            
            valid_range = [c for c in candidates if c['score'] >= (top_score - 0.1)]
            
            # 尝试找单个词
            single_word_winners = [c for c in valid_range if c['is_single']]
            
            if single_word_winners:
                # 找到了高分的单个词 -> 选分数最高的那个单个词
                selected_cand = single_word_winners[0]
                reason = "Single word priority"
            else:
                # 没找到高分单个词 -> 只能选分数最高的词组
                selected_cand = candidates[0]
                reason = "Best match (multi-word)"
            
            print(f"[{original_term}] Selected: '{selected_cand['label']}' ({reason}, Score: {selected_cand['score']})")
            final_results[idx]["type"] = selected_cand['label']

        return final_results

    def _fetch_isa_candidates(self, term):
        candidates = set()
        candidates.update(self._query_db(term))
        
        # 回退机制 (查中心词)
        # 只有当原始词完全没结果时才回退
        if not candidates and " " in term.strip():
            head_word = term.strip().split()[-1]
            print(f"  [Back-off] '{term}' -> '{head_word}'")
            candidates.update(self._query_db(head_word))
            
        return candidates

    def _query_db(self, term):
        found = set()
        norm_term = term.strip().lower().replace(" ", "_")
        
        try:
            label = conceptnet_lite.Label.get(text=norm_term, language='en')
            for concept in label.concepts:
                for edge in concept.edges_out:
                    try:
                        if edge.end.language.name != 'en': continue
                        
                        rel_uri = edge.relation.uri.lower()
                        if '/isa' not in rel_uri and 'is_a' not in rel_uri:
                            # print(f"  [Skipping] Relation '{edge.relation.name}' for term '{term}', uri: {rel_uri}")
                            continue
                        # print(f"  [Found] Relation ({term}, {edge.relation.name}, {edge.end.text})")
                        raw_type = edge.end.text
                        clean_type = self._clean_label(raw_type, term)
                        
                        if clean_type:
                            found.add(clean_type)
                    except: pass
        except conceptnet_lite.peewee.DoesNotExist:
            pass
        except Exception: pass
            
        return found

    def _clean_label(self, text, original_term):
        """
        [温和清洗] 不强制取词尾，只去除无意义前缀
        """
        if not text: return None
        text = text.lower().strip()
        
        # 1. 去除数字和下划线
        text = re.sub(r'^[\d\W_]+', '', text)
        text = text.replace("_", " ")
        
        if text == original_term.lower(): return None
        
        # 2. 去除结构性前缀 (保留后面的整体)
        # 比如 "piece of furniture" -> "furniture" (这里 furniture 是单个词)
        # 比如 "type of programming language" -> "programming language" (这里是词组，保留)
        text = re.sub(r'^(piece|slice|chunk|bit|part|member|state|form|type|kind|sort|variety)\s+of\s+', '', text)
        
        # 3. 去除冠词
        text = re.sub(r'^(a|an|the)\s+', '', text)
        
        # 4. 再次去空格
        text = text.strip()
        
        # 5. 简单的黑名单 (整个词就是 thing 时才过滤，colored thing 不在这里过滤，靠打分)
        if text in ['thing', 'object', 'stuff', 'item', 'something']:
            return None
            
        return text

# ==========================================
# 运行演示
# ==========================================
if __name__ == "__main__":
    tagger = WikidataTagger()
    
    # 示例包含需要用到 MadeOf, ReceivesAction, Has quality 的场景
    batch_input = [
        ("Wooden table", "I bought a sturdy wooden table for the kitchen."), # 测试 MadeOf, Furniture
        ("Apple", "The apple was red and delicious."), # 测试 HasProperty, IsA
        ("Apple", "Apple released a new phone."), # 测试 Company, FieldOfWork
        ("Cake", "The cake was eaten by the children."), # 测试 ReceivesAction,
        ("yellow", "The bright yellow sun shone in the sky."), # 测试 SimilarTo, Color
    ]
    
    results = tagger.batch_process(batch_input)
    
    for (term, context), res in zip(batch_input, results):
        print(f"\nTerm: '{term}' | Context: '{context}'")
        for k, v in res.items():
            print(f"  {k}: {v}")
            
    # tagger_cn = ConceptNetTagger()
    # results_cn = tagger_cn.batch_process(batch_input)
    # for (term, context), res in zip(batch_input, results_cn):
    #     print(f"\n[ConceptNet] Term: '{term}' | Context: '{context}'")
    #     for k, v in res.items():
    #         print(f"  {k}: {v}")