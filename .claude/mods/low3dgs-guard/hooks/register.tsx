/**
 * low3dgs-guard — mod cho dự án nén 3D Gaussian Splatting.
 *
 * Vì sao tồn tại: một loạt thí nghiệm ở đây mất 45–100 phút GPU. Nếu nó chạy khi
 * cây làm việc BẨN thì `run_meta.json` ghi `dirty: true`, và theo đúng quy ước của
 * chính dự án (docs/VERSIONING.md) kết quả đó KHÔNG tái lập được — phát hiện ra
 * sau một tiếng rưỡi thì đã muộn. Trong một phiên làm việc chuyện này suýt xảy ra
 * ba lần.
 *
 * Mod làm hai việc:
 *   1. CHẶN lệnh chạy script thí nghiệm khi cây bẩn hoặc GPU đang bận.
 *   2. Hiện trạng thái dự án trên một dòng ngay trên ô nhập: tag, cây sạch/bẩn,
 *      VRAM, có gì đang chạy không. Im lặng khi mọi thứ ổn, lên màu khi không.
 *
 * Loại trừ `experiments/` khỏi phép kiểm bẩn: chính lần chạy ghi kết quả vào đó,
 * nên nếu tính cả thì không lần chạy nào sạch được. Thứ cần biết là MÃ NGUỒN lúc
 * chạy có khớp commit không.
 */
import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

import type { Health } from '../types'

const health = atom({ plugin: 'low3dgs-guard', key: 'health' } as const, null)

/** Lệnh khởi động một loạt thí nghiệm: scripts/run_*.sh */
const EXPERIMENT = /scripts\/run_[A-Za-z0-9_]+\.sh/
/** Lối thoát có chủ ý, phải gõ rõ ràng */
const OVERRIDE = /ALLOW_DIRTY=1/

const num = (s: string): number | null => {
  const n = Number.parseInt(s.trim(), 10)
  return Number.isNaN(n) ? null : n
}

async function probe($: any): Promise<Health> {
  const at = await $.clock.now()
  let tag: string | null = null
  let dirty: number | null = null
  let gpuMiB: number | null = null
  let busy: number | null = null

  try {
    const r = await $.process.run(['git', 'describe', '--tags', '--abbrev=0'])
    if (r.exitCode === 0) tag = r.stdout.trim() || null
  } catch {}

  try {
    // ':!experiments' — loại trừ thư mục kết quả, xem phần đầu file
    const r = await $.process.run(['git', 'status', '--porcelain', '--', ':!experiments'])
    if (r.exitCode === 0) {
      dirty = r.stdout.split('\n').filter((l: string) => l.trim() !== '').length
    }
  } catch {}

  try {
    const r = await $.process.run(
      ['wsl', 'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
      { timeoutMs: 8000 },
    )
    if (r.exitCode === 0) gpuMiB = num(r.stdout.split('\n')[0] ?? '')
  } catch {}

  try {
    const r = await $.process.run(
      ['wsl', 'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
      { timeoutMs: 8000 },
    )
    if (r.exitCode === 0) {
      busy = r.stdout.split('\n').filter((l: string) => l.trim() !== '').length
    }
  } catch {}

  return { tag, dirty, gpuMiB, busy, at }
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await update($, health, () => null)
    probe($).then(h => update($, health, () => h)).catch(() => {})
    // Làm tươi trong lúc loạt chạy dài đang chạy nền, để dòng trạng thái không đứng im
    $.clock.every(20_000, async () => {
      try {
        const h = await probe($)
        await update($, health, () => h)
      } catch {}
    })
    return next(e)
  })

  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    const cmd = String(e.command ?? '')
    if (!EXPERIMENT.test(cmd) || OVERRIDE.test(cmd)) return next(e)

    const h = await probe($)
    await update($, health, () => h)

    if (h.dirty !== null && h.dirty > 0) {
      return {
        deny:
          `low3dgs-guard: cây làm việc đang BẨN (${h.dirty} mục ngoài experiments/).\n` +
          `Chạy bây giờ thì run_meta.json ghi dirty:true, và theo docs/VERSIONING.md\n` +
          `kết quả KHÔNG tái lập được — loạt này tốn 45-100 phút GPU.\n` +
          `Commit trước đã. Cố ý bỏ qua thì thêm ALLOW_DIRTY=1 vào đầu lệnh.`,
      }
    }
    if (h.busy !== null && h.busy > 0) {
      return {
        deny:
          `low3dgs-guard: GPU đang có ${h.busy} tiến trình chạy (${h.gpuMiB ?? '?'} MiB).\n` +
          `Hai job cùng lúc trên card 4 GB sẽ tranh bộ nhớ và làm hỏng mọi số đo VRAM.\n` +
          `Đợi nó xong, hoặc thêm ALLOW_DIRTY=1 nếu bạn biết mình đang làm gì.`,
      }
    }
    return next(e)
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const h = await read($, health)
    if (h === null || e.props.hasSurvey) return next(e)

    const { Box, Text } = $.ui.resolve(e)
    const isDirty = h.dirty !== null && h.dirty > 0
    const isBusy = h.busy !== null && h.busy > 0

    return (
      <Box>
        <Text dimColor>{h.tag ?? 'low-3DGS'} </Text>
        {isDirty ? (
          <Text color="yellow" bold>
            cây bẩn {h.dirty} mục — commit trước khi chạy{' '}
          </Text>
        ) : (
          <Text dimColor>· cây sạch </Text>
        )}
        {isBusy ? (
          <Text color="cyan" bold>
            · ĐANG CHẠY {h.busy} job · GPU {h.gpuMiB ?? '?'} MiB
          </Text>
        ) : (
          <Text dimColor>· GPU rảnh {h.gpuMiB ?? '?'} MiB</Text>
        )}
      </Box>
    )
  })
}
