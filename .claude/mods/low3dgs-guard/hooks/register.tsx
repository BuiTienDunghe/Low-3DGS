/**
 * low3dgs-guard — mod cho dự án nén 3D Gaussian Splatting.
 *
 * VÌ SAO TỒN TẠI. Một loạt thí nghiệm ở đây tốn 45-100 phút GPU. Chạy khi cây
 * làm việc BẨN thì run_meta.json ghi dirty:true, và theo quy ước của chính dự án
 * (docs/VERSIONING.md) kết quả đó KHÔNG tái lập được — biết sau một tiếng rưỡi
 * thì đã muộn. Trong một phiên, chuyện này suýt xảy ra ba lần.
 *
 * BỐN PHẦN
 *   1. Guard  — chặn lệnh scripts/run_*.sh khi cây bẩn hoặc GPU bận.
 *   2. Band   — một dòng trên ô nhập. LUÔN vẽ gì đó, kể cả lúc chưa đo xong, để
 *               không bao giờ có trường hợp "im lặng nên không biết nó chạy chưa".
 *   3. Pane   — khung riêng: trạng thái dự án + TIẾN TRÌNH loạt đang chạy, đọc từ
 *               log mới nhất. Mở bằng /low3dgs.
 *   4. Nhịp   — ghi kết quả đo ra $.plugin.root/.heartbeat.json để chẩn đoán được
 *               từ bên ngoài. KHÔNG ghi vào repo: làm thế sẽ tự làm bẩn cây, rồi
 *               chính guard này chặn lại.
 */
import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

import type { Health, RunInfo } from '../types'

const PANE = 'low3dgs'
const health = atom({ plugin: 'low3dgs-guard', key: 'health' } as const, null)

const EXPERIMENT = /scripts\/run_[A-Za-z0-9_]+\.sh/
const OVERRIDE = /ALLOW_DIRTY=1/
const PROGRESS = /(\d+)%\|[^|]*\|\s*(\d+)\/(\d+)/

const num = (s: string): number | null => {
  const n = Number.parseInt(s.trim(), 10)
  return Number.isNaN(n) ? null : n
}

const lastMatch = (text: string, re: RegExp): string | null => {
  const all = text.match(re)
  return all === null ? null : all[all.length - 1]
}

const countLines = (s: string): number =>
  s.split('\n').filter(l => l.trim() !== '').length

async function runInfo($: any, at: number): Promise<RunInfo | null> {
  try {
    const entries = await $.fs.list('experiments')
    const logs = entries.filter(
      (f: any) => f.kind === 'file' && f.name.endsWith('.log'),
    )
    if (logs.length === 0) return null
    const newest = logs.reduce((a: any, b: any) => (b.mtimeMs > a.mtimeMs ? b : a))
    const ageSec = Math.round((at - newest.mtimeMs) / 1000)
    // log cũ hơn 3 phút thì coi như không có gì đang chạy
    if (ageSec > 180) return null

    const text = await $.fs.read(`experiments/${newest.name}`)
    const tail = text.slice(-20000)

    let pct: number | null = null
    let step: string | null = null
    const prog = lastMatch(tail, new RegExp(PROGRESS.source, 'g'))
    if (prog !== null) {
      const m = PROGRESS.exec(prog)
      if (m !== null) {
        pct = Number.parseInt(m[1], 10)
        step = `${m[2]}/${m[3]}`
      }
    }
    const psnrLine = lastMatch(tail, /PSNR \d+\.\d+/g)
    const nLine = lastMatch(tail, /Number of Gaussians is \d+/g)

    return {
      name: newest.name,
      pct,
      step,
      psnr: psnrLine === null ? null : psnrLine.replace('PSNR ', ''),
      n: nLine === null ? null : nLine.replace('Number of Gaussians is ', ''),
      ageSec,
    }
  } catch {
    return null
  }
}

async function probe($: any): Promise<Health> {
  const at = await $.clock.now()
  let tag: string | null = null
  let commit: string | null = null
  let dirty: number | null = null
  let gpuMiB: number | null = null
  let busy: number | null = null

  try {
    const r = await $.process.run(['git', 'describe', '--tags', '--abbrev=0'])
    if (r.exitCode === 0) tag = r.stdout.trim() || null
  } catch {}
  try {
    const r = await $.process.run(['git', 'rev-parse', '--short', 'HEAD'])
    if (r.exitCode === 0) commit = r.stdout.trim() || null
  } catch {}
  try {
    // loại trừ experiments/ — chính lần chạy ghi kết quả vào đó, xem đầu file
    const r = await $.process.run([
      'git', 'status', '--porcelain', '--', ':!experiments',
    ])
    if (r.exitCode === 0) dirty = countLines(r.stdout)
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
    if (r.exitCode === 0) busy = countLines(r.stdout)
  } catch {}

  const run = await runInfo($, at)
  const hs: Health = { tag, commit, dirty, gpuMiB, busy, run, at }

  // Nhịp để chẩn đoán từ ngoài. Ghi vào thư mục mod, KHÔNG vào repo.
  try {
    await $.fs.write(`${$.plugin.root}/.heartbeat.json`, JSON.stringify(hs, null, 2))
  } catch {}

  return hs
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    try {
      await $.command.register({
        name: 'low3dgs',
        description: 'Mở khung trạng thái dự án Low-3DGS',
      })
    } catch {}
    probe($)
      .then(hs => update($, health, () => hs))
      .catch(() => {})
    $.clock.every(15_000, async () => {
      try {
        const hs = await probe($)
        await update($, health, () => hs)
      } catch {}
    })
    return next(e)
  })

  on('command.run', { command: 'low3dgs' }, async $ => {
    await $.ui.open({ id: PANE, title: 'Low-3DGS' })
    return { text: 'Đã mở khung trạng thái Low-3DGS.' }
  })

  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    const cmd = String(e.command ?? '')
    if (!EXPERIMENT.test(cmd) || OVERRIDE.test(cmd)) return next(e)

    const hs = await probe($)
    await update($, health, () => hs)

    if (hs.dirty !== null && hs.dirty > 0) {
      return {
        deny:
          `low3dgs-guard: cây làm việc đang BẨN (${hs.dirty} mục ngoài experiments/).\n` +
          `Chạy bây giờ thì run_meta.json ghi dirty:true, và theo docs/VERSIONING.md\n` +
          `kết quả KHÔNG tái lập được — loạt này tốn 45-100 phút GPU.\n` +
          `Commit trước đã. Cố ý bỏ qua thì thêm ALLOW_DIRTY=1 vào đầu lệnh.`,
      }
    }
    if (hs.busy !== null && hs.busy > 0) {
      return {
        deny:
          `low3dgs-guard: GPU đang có ${hs.busy} tiến trình chạy (${hs.gpuMiB ?? '?'} MiB).\n` +
          `Hai job cùng lúc trên card 4 GB sẽ tranh bộ nhớ và làm hỏng mọi số đo VRAM.\n` +
          `Đợi nó xong, hoặc thêm ALLOW_DIRTY=1 nếu bạn biết mình đang làm gì.`,
      }
    }
    return next(e)
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    if (e.props.hasSurvey) return next(e)
    const hs = await read($, health)
    const { Box, Text } = $.ui.resolve(e)

    if (hs === null) {
      return (
        <Box>
          <Text dimColor>low-3DGS · đang đo…</Text>
        </Box>
      )
    }

    const isDirty = hs.dirty !== null && hs.dirty > 0

    return (
      <Box>
        <Text dimColor>{hs.tag ?? 'low-3DGS'} </Text>
        {isDirty ? (
          <Text color="yellow" bold>
            cây bẩn {hs.dirty} mục — commit trước khi chạy{' '}
          </Text>
        ) : (
          <Text dimColor>· sạch </Text>
        )}
        {hs.run !== null ? (
          <Text color="cyan" bold>
            · ĐANG CHẠY {hs.run.pct ?? '?'}% {hs.run.step ?? ''} · GPU {hs.gpuMiB ?? '?'} MiB
          </Text>
        ) : (
          <Text dimColor>· GPU {hs.gpuMiB ?? '?'} MiB</Text>
        )}
      </Box>
    )
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const hs = await read($, health)
    const { Box, Text } = $.ui.resolve(e)

    if (hs === null) {
      return (
        <Box flexDirection="column">
          <Text dimColor>Đang đo lần đầu…</Text>
        </Box>
      )
    }

    const isDirty = hs.dirty !== null && hs.dirty > 0

    return (
      <Box flexDirection="column">
        <Text bold>
          {hs.tag ?? '(chưa có tag)'} · {hs.commit ?? '?'}
        </Text>
        <Text> </Text>

        <Text dimColor>MÃ NGUỒN</Text>
        {isDirty ? (
          <Text color="yellow">
            {'  '}bẩn {hs.dirty} mục — chạy lúc này thì kết quả không truy ngược được
          </Text>
        ) : (
          <Text color="green">{'  '}sạch — run_meta sẽ gắn được commit thật</Text>
        )}
        <Text> </Text>

        <Text dimColor>GPU</Text>
        <Text>
          {'  '}{hs.gpuMiB ?? '?'} MiB / 4096 · trần tensor 3440
        </Text>
        <Text>
          {'  '}{hs.busy === 0 ? 'rảnh' : `${hs.busy ?? '?'} job đang chạy`}
        </Text>
        <Text> </Text>

        <Text dimColor>LOẠT ĐANG CHẠY</Text>
        {hs.run === null ? (
          <Text>{'  '}không có</Text>
        ) : (
          <Box flexDirection="column">
            <Text color="cyan">{'  '}{hs.run.name}</Text>
            <Text>
              {'  '}tiến trình {hs.run.pct ?? '?'}% {hs.run.step ?? ''}
            </Text>
            <Text>{'  '}N = {hs.run.n ?? '?'}</Text>
            <Text>{'  '}PSNR gần nhất {hs.run.psnr ?? '—'}</Text>
            <Text dimColor>{'  '}log ghi cách đây {hs.run.ageSec}s</Text>
          </Box>
        )}
      </Box>
    )
  })
}
