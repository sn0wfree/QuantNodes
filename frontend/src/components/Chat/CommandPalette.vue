<template>
  <a-modal
    v-model:open="visible"
    :footer="null"
    :closable="false"
    :width="500"
    :maskClosable="true"
    wrap-class-name="command-palette-modal"
    @cancel="emit('close')"
  >
    <div class="command-palette">
      <div class="command-search">
        <search-outlined class="search-icon" />
        <input
          v-model="query"
          placeholder="Type a command..."
          ref="searchInput"
          class="search-input"
          @keydown.esc="emit('close')"
          @keydown.up.prevent="moveSelection(-1)"
          @keydown.down.prevent="moveSelection(1)"
          @keydown.enter="executeSelected"
        />
      </div>
      <div class="command-list" v-if="filteredGroups.length">
        <div v-for="group in filteredGroups" :key="group.label" class="command-group">
          <div class="group-label">{{ group.label }}</div>
          <div
            v-for="cmd in group.commands"
            :key="cmd.id"
            class="command-item"
            :class="{ active: selectedId === cmd.id }"
            @click="handleExecute(cmd)"
            @mouseenter="selectedId = cmd.id"
          >
            <span class="command-label">{{ cmd.label }}</span>
            <span class="command-shortcut" v-if="cmd.shortcut">{{ cmd.shortcut }}</span>
          </div>
        </div>
      </div>
      <div class="command-empty" v-else>
        No commands found
      </div>
    </div>
  </a-modal>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted } from 'vue'
import { SearchOutlined } from '@ant-design/icons-vue'
import { useCommands, type Command } from '@/composables/useCommands'

const props = defineProps<{
  open: boolean
}>()

const emit = defineEmits<{
  close: []
}>()

const { commands, execute, search, groups } = useCommands()

const query = ref('')
const selectedId = ref('')
const searchInput = ref<HTMLInputElement>()

const filteredGroups = computed(() => {
  const result = query.value ? search(query.value) : commands.value
  const map = new Map<string, Command[]>()
  for (const cmd of result) {
    if (!map.has(cmd.group)) map.set(cmd.group, [])
    map.get(cmd.group)!.push(cmd)
  }
  return Array.from(map.entries()).map(([label, items]) => ({ label, commands: items }))
})

const flatCommands = computed(() => {
  return filteredGroups.value.flatMap(g => g.commands)
})

const moveSelection = (delta: number) => {
  const list = flatCommands.value
  if (!list.length) return
  const idx = list.findIndex(c => c.id === selectedId.value)
  const next = idx + delta
  if (next < 0) selectedId.value = list[list.length - 1].id
  else if (next >= list.length) selectedId.value = list[0].id
  else selectedId.value = list[next].id
}

const executeSelected = () => {
  const cmd = flatCommands.value.find(c => c.id === selectedId.value)
  if (cmd) handleExecute(cmd)
}

const handleExecute = (cmd: Command) => {
  execute(cmd)
  emit('close')
}

watch(() => props.open, (val) => {
  if (val) {
    query.value = ''
    selectedId.value = flatCommands.value[0]?.id || ''
    nextTick(() => searchInput.value?.focus())
  }
})
</script>

<style scoped>
.command-palette {
  padding: 0;
}

.command-search {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  border-bottom: 1px solid #f0f0f0;
}

.search-icon {
  color: #999;
  font-size: 16px;
}

.search-input {
  flex: 1;
  border: none;
  outline: none;
  font-size: 15px;
  background: transparent;
}

.command-list {
  max-height: 400px;
  overflow-y: auto;
  padding: 8px 0;
}

.command-group {
  padding: 4px 0;
}

.group-label {
  padding: 4px 16px;
  font-size: 12px;
  color: #999;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.command-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 16px;
  cursor: pointer;
  transition: background 0.15s;
}

.command-item:hover,
.command-item.active {
  background: #f5f5f5;
}

.command-label {
  font-size: 14px;
  color: #333;
}

.command-shortcut {
  font-size: 12px;
  color: #999;
  font-family: monospace;
}

.command-empty {
  padding: 24px 16px;
  text-align: center;
  color: #999;
}
</style>

<style>
.command-palette-modal .ant-modal-body {
  padding: 0;
}
</style>
