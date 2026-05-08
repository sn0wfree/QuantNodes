<template>
  <div ref="editorContainer" class="monaco-editor-container" :style="{ height: height + 'px' }"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as monaco from 'monaco-editor'

const props = withDefaults(
  defineProps<{
    modelValue?: string
    language?: string
    height?: number
    readOnly?: boolean
    minimap?: boolean
    fontSize?: number
    theme?: string
  }>(),
  {
    modelValue: '',
    language: 'yaml',
    height: 400,
    readOnly: false,
    minimap: true,
    fontSize: 14,
    theme: 'vs-dark',
  }
)

const emit = defineEmits<{
  'update:modelValue': [value: string]
  save: [value: string]
}>()

const editorContainer = ref<HTMLElement>()
let editor: monaco.editor.IStandaloneCodeEditor | null = null

const initEditor = () => {
  if (!editorContainer.value) return

  editor = monaco.editor.create(editorContainer.value, {
    value: props.modelValue,
    language: props.language,
    theme: props.theme,
    readOnly: props.readOnly,
    minimap: { enabled: props.minimap },
    fontSize: props.fontSize,
    automaticLayout: true,
    scrollBeyondLastLine: false,
    wordWrap: 'on',
    lineNumbers: 'on',
    renderLineHighlight: 'all',
    scrollbar: {
      verticalScrollbarSize: 8,
      horizontalScrollbarSize: 8,
    },
    padding: { top: 10, bottom: 10 },
  })

  editor.onDidChangeModelContent(() => {
    if (editor) {
      emit('update:modelValue', editor.getValue())
    }
  })

  // Add save shortcut (Ctrl+S / Cmd+S)
  editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
    if (editor) {
      emit('save', editor.getValue())
    }
  })
}

watch(
  () => props.modelValue,
  (newValue) => {
    if (editor && editor.getValue() !== newValue) {
      editor.setValue(newValue || '')
    }
  }
)

watch(
  () => props.readOnly,
  (readOnly) => {
    if (editor) {
      editor.updateOptions({ readOnly })
    }
  }
)

onMounted(() => {
  initEditor()
})

onUnmounted(() => {
  if (editor) {
    editor.dispose()
  }
})

// Expose editor methods
defineExpose({
  getEditor: () => editor,
  focus: () => editor?.focus(),
  format: () => {
    editor?.getAction('editor.action.formatDocument')?.run()
  },
})
</script>

<style scoped>
.monaco-editor-container {
  width: 100%;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  overflow: hidden;
}

.monaco-editor-container:focus-within {
  border-color: #1677ff;
  box-shadow: 0 0 0 2px rgba(22, 119, 255, 0.2);
}
</style>
