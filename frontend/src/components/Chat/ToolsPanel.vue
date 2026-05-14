<template>
  <div class="tools-panel">
    <div class="panel-header">
      <span class="panel-title">Tools</span>
      <a-button type="text" size="small" @click="$emit('close')">
        <template #icon><close-outlined /></template>
      </a-button>
    </div>

    <div class="panel-body">
      <div class="section">
        <div class="section-title">Files Changed</div>
        <template v-if="files && files.length">
          <div v-for="f in files" :key="f.path" class="file-item">
            <span :class="'file-status file-' + f.status">
              {{ f.status === 'modified' ? 'M' : f.status === 'created' ? 'A' : 'D' }}
            </span>
            <span class="file-path" :title="f.path">{{ f.path }}</span>
          </div>
        </template>
        <div v-else class="placeholder-text">
          <file-outlined /> File tracking coming in Phase 2
        </div>
      </div>

      <div class="section">
        <div class="section-title">Git History</div>
        <template v-if="commits && commits.length">
          <div v-for="c in commits" :key="c.hash" class="commit-item">
            <span class="commit-hash">{{ c.hash.slice(0, 7) }}</span>
            <span class="commit-msg" :title="c.message">{{ c.message }}</span>
          </div>
        </template>
        <div v-else class="placeholder-text">
          <code-outlined /> Git integration coming in Phase 3
        </div>
      </div>

      <div class="section">
        <div class="section-title">Quick Actions</div>
        <div class="actions-grid">
          <div class="action-btn" @click="$emit('compact')">
            <compress-outlined />
            <span>Compact</span>
          </div>
          <div class="action-btn" @click="$emit('share')">
            <share-alt-outlined />
            <span>Share</span>
          </div>
          <div class="action-btn" @click="$emit('export')">
            <export-outlined />
            <span>Export</span>
          </div>
          <div class="action-btn" @click="$emit('clear')">
            <delete-outlined />
            <span>Clear</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  CloseOutlined,
  FileOutlined,
  CodeOutlined,
  CompressOutlined,
  ShareAltOutlined,
  ExportOutlined,
  DeleteOutlined,
} from '@ant-design/icons-vue'

interface FileChange {
  path: string
  status: 'modified' | 'created' | 'deleted'
}

interface Commit {
  hash: string
  message: string
  timestamp: number
}

defineProps<{
  files?: FileChange[]
  commits?: Commit[]
}>()

defineEmits<{
  close: []
  compact: []
  share: []
  export: []
  clear: []
}>()
</script>

<style scoped>
.tools-panel {
  width: 100%;
  height: 100%;
  background: var(--chat-bg-secondary, #fafafa);
  border-left: 1px solid var(--chat-border-color, #f0f0f0);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  flex-shrink: 0;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 36px;
  padding: 0 12px;
  border-bottom: 1px solid var(--chat-border-color, #f0f0f0);
  flex-shrink: 0;
}

.panel-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--chat-text-primary, #333);
}

.panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}

.section {
  margin-bottom: 16px;
}

.section-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--chat-text-muted, #999);
  margin-bottom: 8px;
}

.file-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 3px 0;
  font-size: 12px;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
}

.file-status {
  width: 16px;
  height: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 700;
  flex-shrink: 0;
}

.file-modified { background: #fff7e6; color: #d46b08; }
.file-created { background: #f6ffed; color: #389e0d; }
.file-deleted { background: #fff1f0; color: #cf1322; }

.file-path {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--chat-text-primary, #333);
}

.commit-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 3px 0;
  font-size: 12px;
}

.commit-hash {
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  color: var(--chat-info, #1677ff);
  flex-shrink: 0;
}

.commit-msg {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--chat-text-secondary, #666);
}

.placeholder-text {
  font-size: 12px;
  color: var(--chat-text-muted, #999);
  padding: 8px 0;
  display: flex;
  align-items: center;
  gap: 6px;
}

.actions-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}

.action-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 8px;
  border: 1px solid var(--chat-border-color, #f0f0f0);
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  color: var(--chat-text-secondary, #666);
  transition: all 0.15s;
}

.action-btn:hover {
  border-color: var(--chat-info, #1677ff);
  color: var(--chat-info, #1677ff);
  background: var(--chat-bg-hover, #f5f5f5);
}
</style>
