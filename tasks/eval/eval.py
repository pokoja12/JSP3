import argparse
import json
from collections import defaultdict

def evaluate(target, gold):
    correct = 0
    total = 0
    tag_type_total = {}
    tag_correct_total = {}
    confusion = defaultdict(lambda: defaultdict(int))

    for group_name in gold["conc"]:
        gold_group = gold["conc"][group_name]
        target_group = target["conc"].get(group_name, {})

        for item_name in gold_group:
            gold_item = gold_group[item_name]
            target_item = target_group.get(item_name)
            if not target_item:
                continue

            tag_g = gold_item.get("tag")
            tag_t = target_item.get("tag")

            if not tag_g or not tag_t:
                continue

            g_key = tag_g.split("-")[-1]
            t_key = tag_t.split("-")[-1]

            total += 1
            tag_type_total[g_key] = tag_type_total.get(g_key, 0) + 1

            if g_key == t_key:
                correct += 1
                tag_correct_total[g_key] = tag_correct_total.get(g_key, 0) + 1

            confusion[g_key][t_key] += 1

    print(f"\n\033[1m---Total accuracy---\033[0m")
    accuracy = correct / total if total > 0 else 0
    print(f"{accuracy:.2%} correct\n")

    print(f"\033[1m---Break down results by tag type---\033[0m")
    for tag in sorted(tag_type_total.keys()):
        correct_type = tag_correct_total.get(tag, 0)
        total_type = tag_type_total[tag]
        accuracy_tag = correct_type / total_type
        print(f"{tag}: {accuracy_tag:.2%} correct ({correct_type}/{total_type})")

    print("\n\033[1m---Confusion matrix---\033[0m")
    all_tags = sorted(tag_type_total.keys())
    print("gold/pred".ljust(10), end="")
    for t in all_tags:
        print(f"{t:>10}", end="")
    print()

    for g in all_tags:
        print(f"{g:<10}", end="")
        for t in all_tags:
            value = confusion[g].get(t, 0)
            if g == t and value > 0:
                print(f"\033[32m{value:>10}\033[0m", end="")
            elif value > 0:
                print(f"\033[31m{value:>10}\033[0m", end="") 
            else:
                print(f"{value:>10}", end="")
        print()

    print("\n\33[1m---F-score---\33[0m")
    print("Precision".rjust(20), end ="")
    print("Recall".rjust(10), end ="")
    print("F1".rjust(10))
    all_tags = sorted(confusion.keys())

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    tags = 0

    for tag in all_tags:
        tp = confusion[tag].get(tag, 0)

        fp = sum(confusion[g].get(tag, 0) for g in all_tags if g != tag)
        fn = sum(confusion[tag].get(t, 0) for t in all_tags if t != tag)

        if tp + fp == 0 or tp + fn == 0:
            continue

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        tags += 1

        print(f"{tag:<5} {precision:>10.3f} {recall:>12.3f} {f1:>13.3f}")

    if tags > 0:
        macro_precision /= tags
        macro_recall /= tags
        macro_f1 /= tags

    print("\n\033[1mMACRO AVG\033[0m")
    print("Precision", end ="")
    print("Recall".rjust(10), end ="")
    print("F1".rjust(10))
    print(f"{macro_precision:.3f} {macro_recall:>12.3f} {macro_f1:>13.3f}")

    print("\n\033[1m---Type of error---\033[0m")


    print("\n\033[1m---Near miss score---\033[0m")


    

def main():
    parser = argparse.ArgumentParser(description="Evaluate model output against gold data")
    parser.add_argument("--target", required=True, help="Path to model output JSON")
    parser.add_argument("--gold", required=True, help="Path to human-annotated JSON")
    args = parser.parse_args()

    with open(args.target, "r", encoding="utf-8") as f:
        target = json.load(f)
    with open(args.gold, "r", encoding="utf-8") as f:
        gold = json.load(f)

    evaluate(target, gold)


if __name__ == "__main__":
    main()
