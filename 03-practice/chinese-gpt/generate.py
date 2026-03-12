"""
中文小说 GPT 文本生成脚本
使用方法：
    python generate.py --model output/model --prompt "第一章"
    python generate.py --model output/model --prompt "第一章" --length 1000 --temperature 0.9
"""

import argparse
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="中文小说 GPT 文本生成")

    # 必需参数
    parser.add_argument("--model", "-m", type=str, required=True, help="模型路径（必需）")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="生成提示文本（必需）")

    # 可选参数
    parser.add_argument("--length", "-l", type=int, default=500, help="生成token数量 (default: 500)")
    parser.add_argument("--temperature", "-t", type=float, default=0.8, help="温度参数，越高越随机 (default: 0.8)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p采样，控制多样性 (default: 0.9)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚 (default: 1.1)")
    parser.add_argument("--device", type=str, default="auto", help="设备：auto/cuda/cpu (default: auto)")

    return parser.parse_args()


def main():
    args = parse_args()

    # 自动选择设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("中文小说 GPT 文本生成")
    print("=" * 60)
    print(f"模型路径: {args.model}")
    print(f"设备: {device}")
    print(f"提示: {args.prompt}")
    print(f"生成长度: {args.length}")
    print(f"温度: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print("=" * 60)

    # 加载模型和分词器
    print("\n加载模型...")
    model = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    print(f"词表大小: {tokenizer.vocab_size}")
    print("模型加载完成！\n")

    # 编码输入
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]

    # 生成文本
    print("=" * 60)
    print("生成结果")
    print("=" * 60)
    print(args.prompt, end="", flush=True)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=prompt_length + args.length,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码并输出
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_part = generated_text[len(args.prompt):]
    print(generated_part)

    print("\n" + "=" * 60)
    print(f"生成完成！总长度: {len(generated_text)} 字符")
    print("=" * 60)


if __name__ == "__main__":
    main()
