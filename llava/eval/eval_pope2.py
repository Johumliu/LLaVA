import os
import re
import json
import argparse
from collections import Counter

def parse_yes_no(text: str) -> str:
    """把自由文本规范化成 'yes' 或 'no'。与原脚本一致的启发式规则。"""
    if text is None:
        return 'yes'  # 保守默认
    t = str(text)
    # 只取第一句
    dot = t.find('.')
    if dot != -1:
        t = t[:dot]
    t = t.replace(',', '').strip()
    words = t.split()
    if 'No' in words or 'not' in words or 'no' in words:
        return 'no'
    return 'yes'

def extract_layer_number_from_path(path: str):
    """
    从文件名中解析层号：优先匹配 *_layer_3*、*layer-2* 等。
    匹配不到返回 None（后续用文件顺序索引兜底）。
    """
    base = os.path.basename(path)
    m = re.search(r"layer[_-]?(\d+)", base, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def load_answers_jsonl(path: str):
    """按原脚本方式加载 answers（每行一个 JSON）。"""
    return [json.loads(q) for q in open(path, 'r', encoding='utf-8')]

def load_questions_map(path: str):
    """加载问题 JSONL，映射 question_id -> question dict。"""
    questions = [json.loads(line) for line in open(path, 'r', encoding='utf-8')]
    return {q['question_id']: q for q in questions}

def load_labels_from_file(label_file: str):
    """POPE 标注文件：每行 JSON，字段 'label' -> 'yes'/'no'"""
    labels = [json.loads(q)['label'] for q in open(label_file, 'r', encoding='utf-8')]
    # 规范成 0/1
    return [0 if (l == 'no') else 1 for l in labels]

def majority_vote_yn(preds_yn):
    """在 ['yes','no'] 上多数投票；平局或空时返回 'no'（保守）。"""
    valid = [p for p in preds_yn if p in ('yes', 'no')]
    if not valid:
        return 'no'
    c = Counter(valid)
    top = c.most_common()
    if len(top) >= 2 and top[0][1] == top[1][1]:
        # 平局时保守返回 'no'
        return 'no'
    return top[0][0]

def eval_pope_ensemble_or(result_files, questions_map, label_file):
    """
    多层 OR 集成评测：
      - result_files: 多个层的结果 jsonl 列表（与层一一对应）
      - questions_map: question_id -> question（用于取 category）
      - label_file: 当前 category 的 label 文件（顺序对齐假设沿用原版）
    说明：
      * 假设各层的结果文件在该 category 下的样本顺序一致（与原脚本相同假设）。
      * 输出与原版相同的指标 + 最早命中层统计。
    """
    # 解析层号（解析失败回退到顺序索引）
    parsed_layers = [extract_layer_number_from_path(p) for p in result_files]
    layer_ids = [n if n is not None else i for i, n in enumerate(parsed_layers)]

    # 加载各层 answers
    layers_answers = [load_answers_jsonl(p) for p in result_files]

    # 取样本数（以第一层为基准）
    N = len(layers_answers[0])

    # 读取标签（与原脚本一致：按文件行序）
    label_list = load_labels_from_file(label_file)  # 0/1
    assert len(label_list) == N, f"标签数({len(label_list)})与答案数({N})不一致：{label_file}"

    # 构建 OR 集成预测 + 最早命中层统计
    pos, neg = 1, 0
    TP = TN = FP = FN = 0
    pred_numeric_list = []  # 用于 Yes ratio 等
    earliest_hit_counter = Counter()  # {layer_id: count}

    for i in range(N):
        # 收集该样本在各层的预测（规范成 yes/no）
        preds_yn = []
        # 也保留原始 'text' 第 0 个元素保持兼容
        for layer_idx in range(len(result_files)):
            ans = layers_answers[layer_idx][i]
            # 原结果结构：answer['text'] 可能是 list，取第一个
            raw_text = ans['text'][0] if isinstance(ans.get('text'), list) else ans.get('text', '')
            preds_yn.append(parse_yes_no(raw_text))

        # 标签
        lbl = label_list[i]            # 0/1
        lbl_yn = 'yes' if lbl == 1 else 'no'

        # 命中层集合（预测等于标签的层）
        hit_layers = {layer_ids[j] for j, p in enumerate(preds_yn) if p == lbl_yn}
        # 最早命中层（按层号最小）
        if hit_layers:
            earliest = min(hit_layers)
            earliest_hit_counter[earliest] += 1

        # 集成预测：OR 命中则直接等于标签；否则用多数投票
        if hit_layers:
            ens_pred_yn = lbl_yn
        else:
            ens_pred_yn = majority_vote_yn(preds_yn)

        ens_pred = 1 if ens_pred_yn == 'yes' else 0
        pred_numeric_list.append(ens_pred)

        # 混淆矩阵四格
        if ens_pred == pos and lbl == pos:
            TP += 1
        elif ens_pred == pos and lbl == neg:
            FP += 1
        elif ens_pred == neg and lbl == neg:
            TN += 1
        elif ens_pred == neg and lbl == pos:
            FN += 1

    # 指标计算（与原版相同，注意 0 除保护）
    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    yes_ratio = pred_numeric_list.count(1) / len(pred_numeric_list) if pred_numeric_list else 0.0

    print('TP\tFP\tTN\tFN\t')
    print(f'{TP}\t{FP}\t{TN}\t{FN}')
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio))
    print('Earliest-hit layer counts (per-question, layer-id as parsed):', dict(sorted(earliest_hit_counter.items())))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    # 改为支持多个层的结果文件
    parser.add_argument("--result-files", type=str, nargs='+', required=True,
                        help="多个层的 POPE 结果 jsonl 文件，文件名最好包含 layer_#")
    args = parser.parse_args()

    questions_map = load_questions_map(args.question_file)

    # 按原脚本：逐类评测（根据问题表的 category 分发样本）
    # 先把每个层的 answers 全量加载，后续再按 category 过滤
    layers_answers = [load_answers_jsonl(p) for p in args.result_files]

    # 为了保持与原版同样的“按类评测”输出，我们还是遍历标注目录中的每个文件
    for file in os.listdir(args.annotation_dir):
        if not (file.startswith('coco_pope_') and file.endswith('.json')):
            continue
        category = file[10:-5]  # 去掉前缀/后缀
        # 逐层：按 category 过滤出该类的答案（顺序保持不变）
        per_layer_cur_answers = []
        for layer_ans in layers_answers:
            cur = [x for x in layer_ans if questions_map[x['question_id']]['category'] == category]
            per_layer_cur_answers.append(cur)

        # 样本数一致性检查（沿用原有顺序对齐假设）
        nset = {len(x) for x in per_layer_cur_answers}
        assert len(nset) == 1, f"各层在类别 {category} 的样本数不一致: {[len(x) for x in per_layer_cur_answers]}"

        print('Category: {}, # samples: {}'.format(category, len(per_layer_cur_answers[0])))

        # 暂存到临时文件的形式不如直接把路径传入 eval 函数更稳妥；这里复用函数，传路径列表即可
        # 为了不改 eval 核心逻辑，这里把 per_layer_cur_answers 写成临时文件也行，
        # 但更简单：为了重用上面的 eval，我们重新实现一个“路径->子集”的传入。
        # 为避免重复实现，这里直接把逻辑嵌入调用：

        # —— 把每层该类子集写入内存并临时模拟路径传入（更直接：再实现一个内联版评估）——
        # 这里直接重用核心评估逻辑（稍微重构一下）：
        # 将 per_layer_cur_answers 拼成“与文件接口一致”的结构临时处理
        # 为保持清晰，我们直接在这里复制 eval 的核心过程：

        # 解析层号
        parsed_layers = [extract_layer_number_from_path(p) for p in args.result_files]
        layer_ids = [n if n is not None else i for i, n in enumerate(parsed_layers)]

        # 标签
        label_file = os.path.join(args.annotation_dir, file)
        label_list = load_labels_from_file(label_file)
        N = len(per_layer_cur_answers[0])
        assert len(label_list) == N, f"标签数({len(label_list)})与类别子集样本数({N})不一致：{label_file}"

        pos, neg = 1, 0
        TP = TN = FP = FN = 0
        pred_numeric_list = []
        earliest_hit_counter = Counter()

        for i in range(N):
            preds_yn = []
            for l in range(len(per_layer_cur_answers)):
                ans = per_layer_cur_answers[l][i]
                raw = ans['text'][0] if isinstance(ans.get('text'), list) else ans.get('text', '')
                preds_yn.append(parse_yes_no(raw))

            lbl = label_list[i]
            lbl_yn = 'yes' if lbl == 1 else 'no'
            hit_layers = {layer_ids[j] for j, p in enumerate(preds_yn) if p == lbl_yn}
            if hit_layers:
                earliest = min(hit_layers)
                earliest_hit_counter[earliest] += 1

            if hit_layers:
                ens_pred_yn = lbl_yn
            else:
                ens_pred_yn = majority_vote_yn(preds_yn)

            ens_pred = 1 if ens_pred_yn == 'yes' else 0
            pred_numeric_list.append(ens_pred)

            if ens_pred == pos and lbl == pos:
                TP += 1
            elif ens_pred == pos and lbl == neg:
                FP += 1
            elif ens_pred == neg and lbl == neg:
                TN += 1
            elif ens_pred == neg and lbl == pos:
                FN += 1

        total = TP + TN + FP + FN
        acc = (TP + TN) / total if total else 0.0
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        yes_ratio = pred_numeric_list.count(1) / len(pred_numeric_list) if pred_numeric_list else 0.0

        print('TP\tFP\tTN\tFN\t')
        print(f'{TP}\t{FP}\t{TN}\t{FN}')
        print('Accuracy: {}'.format(acc))
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F1 score: {}'.format(f1))
        print('Yes ratio: {}'.format(yes_ratio))
        print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio))
        print('Earliest-hit layer counts (per-question, layer-id as parsed):', dict(sorted(earliest_hit_counter.items())))
        print("====================================")

if __name__ == "__main__":
    main()
