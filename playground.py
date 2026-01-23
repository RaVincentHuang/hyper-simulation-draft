
from dataclasses import dataclass
import os
import json
from typing import Callable
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

@dataclass
class CounterInstance:
    query: str
    data: str
    task: str
    query_id: int
    data_id: int
    
    def __str__(self) -> str:
        return f"Task: {self.task} [{self.query_id}:{self.data_id}]\nQuery: {self.query}\nData: {self.data}\n"

class CounterCollector:
    def __init__(self) -> None:
        self.instances: list[CounterInstance] = []
        self.current_query_id: int = -1
    
    def add_counter(self, query: str, data: str, task: str, query_id: int, data_id: int) -> None:
        instance = CounterInstance(
            query=query,
            data=data,
            task=task,
            query_id=query_id,
            data_id=data_id
        )
        self.instances.append(instance)
    
    def display(self) -> None:
        for instance in self.instances:
            print(instance, end="\n---\n")
        
    def save(self, path: str) -> None:
        # save as jsonl
        with open(path, "w") as f:
            for instance in self.instances:
                json.dump(instance.__dict__, f)
                f.write("\n")
    
    def check_at(self, query_id: int) -> 'CounterCollector':
        self.current_query_id = query_id
        return self
    
    @classmethod
    def load(cls, path: str) -> 'CounterCollector':
        collector = cls()
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                instance = CounterInstance(
                    query=data['query'],
                    data=data['data'],
                    task=data['task'],
                    query_id=data['query_id'],
                    data_id=data['data_id']
                )
                collector.instances.append(instance)
        return collector


@dataclass
class Query:
    query: str
    data: list[str]
    consistency: list[bool]
    task: str
    
    @classmethod
    def load_from_json(cls, line: str, task: str):
        data = json.loads(line)
        # data: {q: "...", d: [...], labels: [0, 1, ...]}
        query = data['q']
        d = data['d']
        labels = [bool(x) for x in data['labels']]
        return cls(query=query, data=d, consistency=labels, task=task)
    
    def judge_consistency(self, func: Callable[[str, str], bool], collector: CounterCollector) -> tuple[int, int, float]:
        total = len(self.data)
        inconsistent = 0
        for data_id, (d, label) in enumerate(zip(self.data, self.consistency)):
            pred = func(self.query, d)
            if pred != label:
                collector.add_counter(
                    query=self.query,
                    data=d,
                    task=self.task,
                    query_id=collector.current_query_id,
                    data_id=data_id
                )
                inconsistent += 1
        rate = inconsistent / total if total > 0 else 0.0
        return inconsistent, total, rate

def get_advantage(collector: CounterCollector, baseline_collector: CounterCollector) -> list[CounterInstance]:
    # 找出 collector 中存在但 baseline_collector 中不存在的实例
    # collector 相同 i.i.f : 1. task, 2. query_id, 3. data_id
    baseline_set = set(
        (inst.task, inst.query_id, inst.data_id)
        for inst in baseline_collector.instances
    )
    advantage_instances = [
        inst for inst in collector.instances
        if (inst.task, inst.query_id, inst.data_id) not in baseline_set
    ]
    return advantage_instances
    

# SparseCL
# export SPARSECL_MODEL_PATH="/home/vincent/SparseCL/models"
class SparseCLScorer:
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化 SparseCL 评分器
        :param model_path: 本地模型路径或 HuggingFace 模型 ID
        :param device: 运行设备 ('cuda' 或 'cpu')
        """
        self.device = device
        print(f"正在加载模型: {model_path} 到 {self.device}...")
        
        # 加载 Tokenizer 和 Model
        # 注意：trust_remote_code=True 是必须的，因为 SparseCL 可能使用了自定义模型结构
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.model.eval() # 切换到评估模式

    def _mean_pooling(self, model_output, attention_mask):
        """
        执行平均池化 (Mean Pooling)，这是论文中指定的池化方式 
        """
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, text: str):
        """
        获取单个文本的嵌入向量
        """
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 论文指出在使用微调模型时应使用 "avg" pooler 
        embedding = self._mean_pooling(outputs, inputs['attention_mask'])
        return F.normalize(embedding, p=2, dim=1) # 规范化向量

    def _calc_hoyer_sparsity(self, v1: torch.Tensor, v2: torch.Tensor):
        """
        计算两个向量差值的 Hoyer 稀疏度
        公式: (sqrt(d) - L1/L2) / (sqrt(d) - 1) [cite: 33, 160]
        """
        # 1. 计算差值向量
        diff = v1 - v2
        
        # 2. 获取维度 d
        d = diff.shape[1]
        sqrt_d = torch.sqrt(torch.tensor(d, device=self.device))
        
        # 3. 计算 L1 和 L2 范数
        l1_norm = torch.norm(diff, p=1, dim=1)
        l2_norm = torch.norm(diff, p=2, dim=1)
        
        # 4. 应用 Hoyer 公式
        # 防止除以零 (虽然在嵌入差值中不太可能发生)
        l2_norm = torch.clamp(l2_norm, min=1e-9)
        
        hoyer = (sqrt_d - (l1_norm / l2_norm)) / (sqrt_d - 1)
        return hoyer

    def compute_score(self, text_a: str, text_b: str, alpha: float = 1.0) -> float:
        """
        计算两个文本的矛盾得分
        :param text_a: 文本 A (Query)
        :param text_b: 文本 B (Document)
        :param alpha: 稀疏度权重的超参数，需根据数据调整 
        :return: 综合得分 (Cosine + alpha * Hoyer)
        """
        emb_a = self.get_embedding(text_a)
        emb_b = self.get_embedding(text_b)

        # 1. 计算余弦相似度 (Cosine Similarity) [cite: 171]
        # 因为向量已经 normalize 过了，所以点积即为余弦相似度
        cosine_score = torch.sum(emb_a * emb_b, dim=1)

        # 2. 计算 Hoyer 稀疏度 (Hoyer Sparsity) [cite: 160, 171]
        hoyer_score = self._calc_hoyer_sparsity(emb_a, emb_b)

        # 3. 加权求和
        final_score = cosine_score + alpha * hoyer_score
        
        return final_score.item()

def load_sparsecl_scorer() -> SparseCLScorer:
    model_path = os.environ.get("SPARSECL_MODEL_PATH", "SparseCL/sparsecl-base-uncased")
    # {model_path}/GTE-SparseCL-msmarco/
    path = os.path.join(model_path, "GTE-SparseCL-msmarco")
    scorer = SparseCLScorer(model_path=path)
    return scorer

def consistency_sparsecl(query: str, data: str, scorer: SparseCLScorer, threshold: float = 0.5) -> bool:
    """
    使用 SparseCL 评分器判断 query 和 data 是否一致
    :param query: 查询文本
    :param data: 文档文本
    :param scorer: SparseCL 评分器实例
    :param threshold: 判定一致性的阈值
    :return: 一致性布尔值
    """
    score = scorer.compute_score(query, data)
    return score >= threshold

def get_sparsecl_threshold(path: str, scorer: SparseCLScorer) -> float:
    consistent_scores = []
    inconsistent_scores = []
    with open(path, "r") as f:
        for i, line in tqdm(enumerate(f), desc="Processing queries"):
            q = Query.load_from_json(line, task)
            # calc the scores of consistent and inconsistent pairs
            for d, label in zip(q.data, q.consistency):
                score = scorer.compute_score(q.query, d)
                if label:
                    consistent_scores.append(score)
                else:
                    inconsistent_scores.append(score)
    
    # Calculate threshold based on scores
    
    # AVG of consistent scores and inconsistent scores
    avg_consistent = sum(consistent_scores) / len(consistent_scores) if consistent_scores else 0.0
    avg_inconsistent = sum(inconsistent_scores) / len(inconsistent_scores) if inconsistent_scores else 0.0
    print(f"Average Consistent Score: {avg_consistent:.4f}")
    print(f"Average Inconsistent Score: {avg_inconsistent:.4f}")
    
    # MAX and MIN of consistent and inconsistent scores
    max_inconsistent = max(inconsistent_scores) if inconsistent_scores else 0.0
    min_consistent = min(consistent_scores) if consistent_scores else 0.0
    print(f"Max Inconsistent Score: {max_inconsistent:.4f}")
    print(f"Min Consistent Score: {min_consistent:.4f}")
    
    return 0.5

QUERY_DATASET_PATH = os.environ.get("QUERY_DATASET", "data/queries.jsonl")

TASKS = ["RAG/ARC", "RAG/popQA"]

if __name__ == "__main__":
    
    collector = CounterCollector()
    
    # 1. SparseCL
    scorer = load_sparsecl_scorer()
    sparsecl_threshold = 0.5
    
    for task in TASKS:
        path = os.path.join(QUERY_DATASET_PATH, f"{task}.jsonl")
        get_sparsecl_threshold(path, scorer)
        exit(0)
        total_inconsistent = 0
        total_count = 0
        with open(path, "r") as f:
            for i, line in tqdm(enumerate(f), desc=f"Judging consistency for {task}"):
                q = Query.load_from_json(line, task)
                inconsistent, count, rate = q.judge_consistency(
                    lambda query, data: consistency_sparsecl(query, data, scorer, threshold=sparsecl_threshold),
                    collector.check_at(i)
                )
                total_inconsistent += inconsistent
                total_count += count
        overall_rate = total_inconsistent / total_count if total_count > 0 else 0.0
        print(f"Task: {task}, Inconsistent: {total_inconsistent}/{total_count}, Rate: {overall_rate:.4f}")