import { get, post } from './index'

export interface ChatMessage {
  content: string
  session_id?: string
}

export interface ChatResponse {
  message_id: string
  content: string
  tools_used: string[]
  usage: Record<string, number>
}

export interface ToolCallInfo {
  tool_name: string
  arguments: Record<string, any>
  result?: Record<string, any>
}

export const agentApi = {
  sendMessage: (data: ChatMessage) =>
    post<ChatResponse>('/chat', data),

  getHistory: (sessionId: string) =>
    get<ChatMessage[]>(`/chat/history/${sessionId}`),
}
