# Đo đỉnh VRAM TỪ TRONG TIẾN TRÌNH — trả nợ EXP-015 kết luận 3.
# Đặt ngoài repo, nạp qua PYTHONPATH => 0 dòng sửa trong mã bên thứ ba.
# Phải dùng atexit: prune_finetune.py có  ngay trong vòng lặp
# (if iteration < opt.iterations: step ... else: return), nên MỌI mã đặt sau
# vòng lặp đều không bao giờ chạy.
import atexit, sys

def _dump():
    try:
        t = sys.modules.get("torch")
        if t is None or not t.cuda.is_available():
            return
        print("[MEMPROBE] max_allocated_MiB=%.1f max_reserved_MiB=%.1f"
              % (t.cuda.max_memory_allocated() / 2**20,
                 t.cuda.max_memory_reserved() / 2**20), flush=True)
    except Exception as e:
        print("[MEMPROBE] failed: %r" % (e,), flush=True)

atexit.register(_dump)
