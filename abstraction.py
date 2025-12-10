import spacy
import nltk
from nltk.corpus import wordnet as wn
from pywsd.lesk import simple_lesk
from spacy.tokens import Token


INDUSTRIAL_ANCHORS = [
    # --- IT & Data (数字资产) ---
    ("ProgrammingLanguage", {"programming_language.n.01"}),
    ("Software",            {"software.n.01", "computer_program.n.01", "app.n.01", "operating_system.n.01"}),
    ("Database",            {"database.n.01", "information_system.n.01"}),
    ("Network",             {"computer_network.n.01", "internet.n.01", "web.n.01", "cyberspace.n.01"}),
    ("Hardware",            {"computer_hardware.n.01", "hardware.n.03", "device.n.01", "server.n.03"}),
    ("CyberAttack",         {"cyberattack.n.01", "virus.n.03", "malware.n.01"}),
    ("Data",                {"data.n.01", "information.n.01", "dataset.n.01", "record.n.01", "file.n.01"}),
    
    # --- Transport & Logistics (交通物流) ---
    ("Aircraft",            {"aircraft.n.01", "drone.n.01"}),
    ("Watercraft",          {"vessel.n.03", "ship.n.01", "boat.n.01"}),
    ("RailVehicle",         {"train.n.01", "locomotive.n.01"}),
    ("Vehicle",             {"vehicle.n.01", "transport.n.01", "car.n.01", "truck.n.01", "automobile.n.01"}), 
    ("Infrastructure",      {"infrastructure.n.02", "road.n.01", "bridge.n.01", "airport.n.01", "port.n.01"}),

    # --- Manufacturing & Engineering (制造工程) ---
    ("Factory",             {"factory.n.01", "plant.n.01", "manufacturing_plant.n.01"}),
    ("Machine",             {"machine.n.01", "engine.n.01", "motor.n.01", "robot.n.01", "mechanism.n.05"}),
    ("Tool",                {"tool.n.01", "implement.n.01"}),
    ("Material",            {"material.n.01", "raw_material.n.01", "substance.n.01", "metal.n.01", "plastic.n.01"}),
    ("Energy",              {"energy.n.01", "electricity.n.01", "power.n.01", "fuel.n.01"}),

    # --- Business & Legal (商业法律) ---
    ("Company",             {"company.n.01", "enterprise.n.02", "corporation.n.01", "firm.n.01"}),
    ("Government",          {"government.n.01", "agency.n.01", "authority.n.02"}),
    ("Team",                {"team.n.01", "committee.n.01", "commission.n.01"}),
    ("Contract",            {"contract.n.01", "agreement.n.01", "deal.n.01"}),
    ("Regulation",          {"law.n.02", "regulation.n.01", "rule.n.03", "policy.n.02", "standard.n.01"}),
    ("Document",            {"document.n.01", "report.n.01", "publication.n.01", "manual.n.01"}),
    ("Currency",            {"currency.n.01", "money.n.01"}),

    # --- People & Roles (人物角色) ---
    ("Professional",        {"professional.n.01", "expert.n.01", "engineer.n.01", "scientist.n.01", "manager.n.01", "developer.n.01"}),
    ("Worker",              {"worker.n.01", "employee.n.01"}),
    ("Customer",            {"customer.n.01", "client.n.01", "user.n.01"}),
    ("Leader",              {"leader.n.01", "head.n.01", "director.n.01", "chief.n.01"}),
    ("Person",              {"person.n.01"}),

    # --- Location (地点) ---
    ("City",                {"city.n.01", "town.n.01", "municipality.n.01"}),
    ("Country",             {"country.n.02", "nation.n.02"}),
    ("Building",            {"building.n.01", "structure.n.01", "edifice.n.01"}),
    ("Room",                {"room.n.01", "area.n.05"}),
    ("NaturalPlace",        {"body_of_water.n.01", "geological_formation.n.01"}),
    
    # --- Activities & Events (动态) ---
    ("Project",             {"project.n.01", "undertaking.n.01", "program.n.02"}),
    ("Process",             {"process.n.06", "procedure.n.01", "workflow.n.01", "operation.n.05"}),
    ("Transaction",         {"transaction.n.01", "sale.n.01", "payment.n.01"}),
    ("Incident",            {"accident.n.01", "error.n.01", "failure.n.01", "crash.n.02", "bug.n.02"}),
    ("Meeting",             {"meeting.n.01", "conference.n.01", "gathering.n.01"}),
    
    # --- Concepts (抽象) ---
    ("Metric",              {"measure.n.02", "quantity.n.01", "rate.n.02", "amount.n.03", "value.n.02"}),
    ("TimePeriod",          {"time_period.n.01", "time_unit.n.01"}),
    ("Method",              {"method.n.01", "technique.n.01", "algorithm.n.01", "approach.n.01"}),
    ("Problem",             {"problem.n.01", "trouble.n.01", "issue.n.01"}),
    
    # --- Fallbacks (通用兜底) ---
    ("Group",               {"group.n.01", "organization.n.01", "social_group.n.01"}),
    ("Location",            {"location.n.01", "place.n.01", "area.n.01"}),
    ("Event",               {"event.n.01"}),
    ("Object",              {"artifact.n.01", "physical_entity.n.01", "object.n.01"}),
    ("Concept",             {"abstraction.n.06", "idea.n.01"})
]

class TokenAbstractor:
    def __init__(self):
        self.anchors = INDUSTRIAL_ANCHORS
        # 为了加速匹配，可以将 anchor set 预处理，但这里为了逻辑清晰保持原样

    def _spacy_to_wn_pos(self, spacy_pos):
        """将 SpaCy POS 转换为 WordNet POS"""
        if spacy_pos in ['NOUN', 'PROPN']: return wn.NOUN
        if spacy_pos == 'VERB': return wn.VERB
        if spacy_pos == 'ADJ': return wn.ADJ
        if spacy_pos == 'ADV': return wn.ADV
        return None
    
    def _get_contextual_synset(self, token: Token, doc):
        """
        核心逻辑：使用 PyWSD 进行词义消歧
        """
        wn_pos = self._spacy_to_wn_pos(token.pos_)
        if not wn_pos:
            return None

        # A. 尝试使用 Lesk 算法结合上下文 (Doc文本)
        try:
            # simple_lesk(句子文本, 目标词, 词性)
            synset = simple_lesk(doc.text, token.text, pos=wn_pos)
        except Exception:
            synset = None

        # B. 兜底机制：如果 Lesk 失败 (比如句子太短无重叠词)，回退到 MFS (最常用义项)
        if not synset:
            synsets = wn.synsets(token.lemma_, pos=wn_pos)
            if synsets:
                synset = synsets[0]
        
        return synset

    def get_abstraction(self, token: Token, doc) -> str:
        """
        输入 Token 和上下文 Doc，返回类型
        """
        # 1. 代词处理
        if token.pos_ == "PRON":
            if token.lower_ in ["he", "she", "who", "i"]: return "Person"
            return "Object"

        # 2. 获取经过消歧的 Synset
        target_synset = None
        if token.pos_ in ["NOUN", "PROPN"]:
            target_synset = self._get_contextual_synset(token, doc)
        
        wn_cat = None
        if target_synset:
            # 获取路径并匹配锚点
            hypernyms = set()
            for path in target_synset.hypernym_paths():
                for node in path:
                    hypernyms.add(node.name())
            
            # 匹配锚点
            for category, anchor_set in self.anchors:
                if not hypernyms.isdisjoint(anchor_set):
                    wn_cat = category
                    break # 找到最高优先级的分类即停止

        ner_tag = token.ent_type_

        # 3. 混合决策 (NER + WSD)
        if ner_tag:
            if ner_tag == "ORG":
                if wn_cat in ["Company", "FinancialInst"]: return wn_cat
                return "Company" # Default refinement
            
            if ner_tag == "GPE" or ner_tag == "LOC":
                if wn_cat in ["NaturalPlace", "City", "Country"]: return wn_cat
                return "Location"
            
            return "Entity" # 简化处理其他 NER

        # 4. 纯 WordNet 结果
        if wn_cat:
            return wn_cat
            
        # 5. 最终兜底
        return "Object" if token.pos_ in ["NOUN", "PROPN"] else "Unknown"

# ==============================================================================
# 4. 测试与演示
# ==============================================================================
if __name__ == "__main__":
    abstractor = TokenAbstractor()
    nlp = spacy.load('en_core_web_trf')
    # 定义不同工业领域的测试句
    test_cases = [
        ("IT/Tech", "The Python script crashed the server due to a memory leak."),
        ("Business", "The CEO of Google signed the new contract yesterday."),
        ("Logistics", "The freight truck delivered the containers to the warehouse."),
        ("Manufacturing", "The engineer checked the viscosity of the oil in the engine."),
        ("Finance", "The transaction fee was 5 dollars per user.")
    ]

    print(f"{'Token':<15} | {'Lemma':<12} | {'NER':<8} | {'Abstract Type (Result)'}")
    print("-" * 65)

    for domain, text in test_cases:
        print(f"--- Domain: {domain} ---")
        doc = nlp(text)
        
        # 仅展示名词性成分的抽象结果
        processed_tokens = []
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                abstract_type = abstractor.get_abstraction(token, doc)
                print(f"{token.text:<15} | {token.lemma_:<12} | {token.ent_type_:<8} | {abstract_type}")
        print("")