"""
诡秘之主 GPT 文本生成脚本
使用训练好的模型生成小说风格的中文文本
"""

import os
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


class GuimiGenerator:
    def __init__(self, model_dir="./output/model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("加载模型...")

        # 加载tokenizer
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(model_dir, "tokenizer.json"),
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
        )

        # 加载模型
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"设备: {self.device}")
        print(f"词表大小: {self.tokenizer.vocab_size}")
        print("模型加载完成!\n")

    def generate(self, prompt, max_length=200, temperature=0.8, top_k=50, top_p=0.95):
        """生成文本"""
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # 生成
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )

        # 解码
        tokens = self.tokenizer.convert_ids_to_tokens(output_ids[0])
        output_text = self.tokenizer.convert_tokens_to_string(tokens)
        output_text = output_text.replace(" ", "").replace("▁", "")  # 去除空格和tokenizer前缀

        return output_text


def demo():
    """演示生成"""
    print("=" * 60)
    print("诡秘之主 GPT 文本生成")
    print("=" * 60)

    generator = GuimiGenerator()

    # 测试提示
    prompts = [
        "克莱恩看着面前的笔记本",
        "绯红的月光照耀着廷根市",
        "周明瑞醒来后发现自己",
        "非凡者世界的秘密在于",
    ]

    print("\n生成示例:")
    print("=" * 60)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}] 提示: {prompt}")
        print("-" * 40)

        output = generator.generate(prompt, max_length=150)
        print(output)
        print()


def interactive():
    """交互模式"""
    print("=" * 60)
    print("诡秘之主 GPT 交互模式")
    print("输入提示文本，模型将续写故事")
    print("输入 'quit' 退出")
    print("=" * 60)

    generator = GuimiGenerator()

    while True:
        try:
            prompt = input("\n请输入提示: ").strip()

            if prompt.lower() == "quit":
                print("再见!")
                break

            if not prompt:
                continue

            print("\n生成中...")
            output = generator.generate(prompt, max_length=200)

            print("\n" + "-" * 40)
            print(output)
            print("-" * 40)

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}")


def main():
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "-i":
            interactive()
        else:
            # 单次生成
            generator = GuimiGenerator()
            prompt = sys.argv[1]
            max_length = int(sys.argv[2]) if len(sys.argv) > 2 else 200

            output = generator.generate(prompt, max_length=max_length)
            print(output)
    else:
        demo()


if __name__ == "__main__":
    main()
