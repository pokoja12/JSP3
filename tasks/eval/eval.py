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
         
    accuracy = correct / total if total > 0 else 0

    print(f"\n---Total accuracy---")
    print(f"{accuracy:.2%} correct\n")

    print(f"---Break down results by tag type---")
    for tag in sorted(tag_type_total.keys()):
        correct_type = tag_correct_total.get(tag, 0)
        total_type = tag_type_total[tag]
        accuracy_tag = correct_type / total_type
        print(f"{tag}: {accuracy_tag:.2%} correct ({correct_type}/{total_type})")

    print("\n---Confusion matrix---")
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

    print("\n---type of error---")


    print("\n---Near miss score---")


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
