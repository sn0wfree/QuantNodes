import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: () => import('@/components/Layout/AppLayout.vue'),
    children: [
      {
        path: '',
        name: 'Dashboard',
        component: () => import('@/views/Dashboard/index.vue'),
        meta: { title: 'Dashboard' },
      },
      {
        path: 'wiki/factors',
        name: 'FactorList',
        component: () => import('@/views/Wiki/FactorList.vue'),
        meta: { title: 'Factors' },
      },
      {
        path: 'wiki/factors/:name',
        name: 'FactorDetail',
        component: () => import('@/views/Wiki/FactorDetail.vue'),
        meta: { title: 'Factor Detail' },
      },
      {
        path: 'wiki/strategies',
        name: 'StrategyList',
        component: () => import('@/views/Wiki/StrategyList.vue'),
        meta: { title: 'Strategies' },
      },
      {
        path: 'wiki/strategies/:name',
        name: 'StrategyDetail',
        component: () => import('@/views/Wiki/StrategyDetail.vue'),
        meta: { title: 'Strategy Detail' },
      },
      {
        path: 'backtest',
        name: 'BacktestCenter',
        component: () => import('@/views/Backtest/ConfigEditor.vue'),
        meta: { title: 'Backtest Center' },
      },
      {
        path: 'backtest/result/:id',
        name: 'BacktestResult',
        component: () => import('@/views/Backtest/ResultView.vue'),
        meta: { title: 'Backtest Result' },
      },
      {
        path: 'strategy/editor',
        name: 'StrategyEditor',
        component: () => import('@/views/Strategy/Editor.vue'),
        meta: { title: 'Strategy Editor' },
      },
      {
        path: 'factor-analysis',
        name: 'FactorAnalysis',
        component: () => import('@/views/FactorAnalysis/index.vue'),
        meta: { title: 'Factor Analysis' },
      },
      {
        path: 'dream',
        name: 'DreamInsights',
        component: () => import('@/views/Dream/index.vue'),
        meta: { title: 'Dream Insights' },
      },
      {
        path: 'settings',
        name: 'Settings',
        component: () => import('@/views/Settings/index.vue'),
        meta: { title: 'Settings' },
      },
      {
        path: 'chat',
        name: 'AgentChat',
        component: () => import('@/views/AgentChat/index.vue'),
        meta: { title: 'Agent Chat' },
      },
    ],
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('@/views/NotFound.vue'),
    meta: { title: '404' },
  },
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
  scrollBehavior() {
    return { top: 0 }
  },
})

router.beforeEach((to, _from, next) => {
  document.title = `${to.meta.title || 'QuantNodes'} - QuantNodes`
  next()
})

export default router
