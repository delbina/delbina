"""Minimal offline demo: compress a long synthetic chat and print stats."""
from delbina import Compressor

history = [{"role": "system", "content": "You are a helpful assistant."}]
for i in range(40):
    history.append({"role": "user", "content": f"Question {i}: " + "details " * 30})
    history.append({"role": "assistant", "content": f"Answer {i}: " + "explanation " * 60})

comp = Compressor(max_tokens=1500, keep_last=4)
small = comp.compress(history)

print(f"messages: {len(history)} -> {len(small)}")
print(comp.last_stats, f"saved={comp.last_stats.saved:.0%}")
