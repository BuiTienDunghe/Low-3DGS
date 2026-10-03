export type Health = {
  /** tag gần nhất, ví dụ "v0.9.0"; null nếu không đọc được */
  tag: string | null
  /** số mục bẩn NGOÀI experiments/ — chính lần chạy ghi vào đó nên phải loại trừ */
  dirty: number | null
  /** VRAM đang dùng, MiB */
  gpuMiB: number | null
  /** số tiến trình đang chiếm GPU */
  busy: number | null
  /** thời điểm đo, ms từ epoch */
  at: number
}

declare module 'claude-code' {
  interface PluginState {
    'low3dgs-guard': { health: Health | null }
  }
}
