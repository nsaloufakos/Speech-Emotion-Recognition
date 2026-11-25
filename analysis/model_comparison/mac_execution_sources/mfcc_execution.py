"""Equivalent dense execution of the existing packed one-layer BiLSTM.

Right padding preserves forward states; left padding preserves backward states.
Selecting the last/first valid positions gives the same representation as packing.
This enables fused MPS LSTM kernels without changing learned parameters.
"""

import argparse
import time

import torch
from models.baseline_mfcc_bilstm import BiLSTM


class DenseBiLSTM(BiLSTM):
    def forward(self, audio, lengths):
        batch, frames, features = audio.shape
        lengths = lengths.to(audio.device)
        positions = torch.arange(frames, device=audio.device)[None, :]
        source = positions - (frames - lengths[:, None])
        left = audio.gather(1, source.clamp(min=0).unsqueeze(-1).expand(-1, -1, features))
        left = left * (source >= 0).unsqueeze(-1).to(audio.dtype)
        output, _ = self.lstm(torch.cat([audio, left], dim=0))
        rows = torch.arange(batch, device=audio.device)
        hidden = self.lstm.hidden_size
        forward = output[rows, lengths - 1, :hidden]
        backward = output[batch + rows, frames - lengths, hidden:]
        return self.classifier(torch.cat([forward, backward], dim=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(473)
    reference = BiLSTM()
    candidate = DenseBiLSTM().to(args.device)
    candidate.load_state_dict(reference.state_dict())
    lengths = torch.tensor([1, 2, 7, 15, 16, 31, 45, 64])
    audio = torch.randn(8, 64, 40)
    for i, length in enumerate(lengths):
        audio[i, length:] = 0
    labels = torch.arange(8)
    target = reference(audio, lengths)
    result = candidate(audio.to(args.device), lengths)
    output_error = float((target - result.cpu()).abs().max())
    torch.nn.functional.cross_entropy(target, labels).backward()
    torch.nn.functional.cross_entropy(result, labels.to(args.device)).backward()
    gradient_error = max(float((p.grad - dict(candidate.named_parameters())[name].grad.cpu()).abs().max())
                         for name, p in reference.named_parameters())
    if output_error > 2e-4 or gradient_error > 2e-4:
        raise AssertionError(f"Packed/dense mismatch: outputs={output_error}, gradients={gradient_error}")
    print(f"Equivalent logits/parameter gradients: max errors {output_error:.3g}/{gradient_error:.3g}", flush=True)
    model = DenseBiLSTM().to(args.device)
    data = torch.randn(32, 128, 40, device=args.device)
    lengths = torch.arange(97, 129)
    labels = (torch.arange(32) % 8).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for step in range(24):
        if step == 4:
            if args.device == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(data, lengths), labels)
        loss.backward()
        optimizer.step()
    if args.device == "mps":
        torch.mps.synchronize()
    print(f"{args.device}: {(time.perf_counter() - start) / 20:.4f} seconds per training batch", flush=True)


if __name__ == "__main__":
    main()
