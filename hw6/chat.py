import torch
from tokenizers import Tokenizer
from generator import GeneratorTransformer

def chat(checkpoint_path, tokenizer_path="transformer_basics/mistral_tokenizer.json", context_len=64, use_beam=False):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.add_special_tokens(["<s>", "</s>", "<pad>"])

    model = GeneratorTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=256,
        num_heads=8,
        d_ff=512,
        num_layers=4,
        pad_token_id=tokenizer.token_to_id("<pad>"),
        bos_token_id=tokenizer.token_to_id("<s>"),
        eos_token_id=tokenizer.token_to_id("</s>"),
        max_len=context_len,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


    model.load_state_dict(torch.load(checkpoint_path, map_location=model.device))
    model.eval()

    print("Chat mode")
    print(f"Режим генерации: {'Beam' if use_beam else 'Stock'}")
    while True:
        user_input = input("Вы: ")
        if user_input.lower() in {"quit", "exit"}:
            break

        if use_beam:
            output = model.generate_beam(user_input, context_len=context_len, max_out_tokens=100, beam_width=5)
        else:
            output = model.generate(user_input, context_len=context_len, max_out_tokens=100, temperature=0.8)

        output = output.replace("<pad>", "").replace("<s>", "").replace("</s>", "")
        output = output.replace("\n", " ").replace("  ", " ").strip()
        print(f"Бот: {output.strip()}")

chat("generator.pt_epoch7.pt", use_beam=False)
