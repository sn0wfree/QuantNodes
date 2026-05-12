import { ref } from 'vue'

export interface Command {
  id: string
  label: string
  group: string
  shortcut?: string
  icon?: string
  action: () => void | Promise<void>
}

const commands = ref<Command[]>([])

export function useCommands() {
  const register = (cmd: Command) => {
    const existing = commands.value.find(c => c.id === cmd.id)
    if (!existing) {
      commands.value.push(cmd)
    }
  }

  const unregister = (id: string) => {
    commands.value = commands.value.filter(c => c.id !== id)
  }

  const execute = (cmd: Command) => {
    cmd.action()
  }

  const search = (query: string) => {
    if (!query.trim()) return commands.value
    const lower = query.toLowerCase()
    return commands.value.filter(
      c => c.label.toLowerCase().includes(lower) || c.group.toLowerCase().includes(lower)
    )
  }

  const groups = () => {
    const map = new Map<string, Command[]>()
    for (const cmd of commands.value) {
      if (!map.has(cmd.group)) map.set(cmd.group, [])
      map.get(cmd.group)!.push(cmd)
    }
    return Array.from(map.entries()).map(([label, items]) => ({ label, commands: items }))
  }

  return { commands, register, unregister, execute, search, groups }
}
