/** Một loạt chạy đang diễn ra, đọc từ log mới nhất trong experiments/ */
export type RunInfo = {
  /** tên file log, ví dụ "exp026_s2attr_p010.log" */
  name: string
  /** phần trăm tiến trình huấn luyện, null nếu chưa thấy dòng progress */
  pct: number | null
  /** "2000/5000" */
  step: string | null
  /** PSNR của lần eval gần nhất */
  psnr: string | null
  /** số Gaussian gần nhất log in ra */
  n: string | null
  /** log được ghi cách đây bao nhiêu giây */
  ageSec: number
}

export type Health = {
  /** tag gần nhất, ví dụ "v0.9.1" */
  tag: string | null
  /** commit ngắn */
  commit: string | null
  /** số mục bẩn NGOÀI experiments/ — chính lần chạy ghi vào đó nên phải loại trừ */
  dirty: number | null
  /** VRAM đang dùng, MiB */
  gpuMiB: number | null
  /** số tiến trình đang chiếm GPU */
  busy: number | null
  /** loạt chạy đang diễn ra, null nếu không có */
  run: RunInfo | null
  /** thời điểm đo, ms từ epoch */
  at: number
}

declare module 'claude-code' {
  interface PluginState {
    'low3dgs-guard': { health: Health | null }
  }
}
