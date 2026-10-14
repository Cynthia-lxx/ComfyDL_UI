/*!
 * Live2D Widget (autoload) — adapted for the ComfyDL_UI homepage
 * https://github.com/stevenjoezhang/live2d-widget
 *
 * Differences from upstream autoload.js:
 *  - skipped entirely for prefers-reduced-motion users (decorative motion)
 *  - a small style override re-docks #waifu to the bottom-right, because the
 *    bottom-left corner hosts this page's sidebar icon rail
 */

// Recommended to use absolute path for live2d_path parameter
// live2d_path 参数建议使用绝对路径
const live2d_path = 'https://fastly.jsdelivr.net/npm/live2d-widgets@1.0.0-rc.6/dist/';
// const live2d_path = '/dist/';

// Method to encapsulate asynchronous resource loading
// 封装异步加载资源的方法
function loadExternalResource(url, type) {
  return new Promise((resolve, reject) => {
    let tag;

    if (type === 'css') {
      tag = document.createElement('link');
      tag.rel = 'stylesheet';
      tag.href = url;
    }
    else if (type === 'js') {
      tag = document.createElement('script');
      tag.type = 'module';
      tag.src = url;
    }
    if (tag) {
      tag.onload = () => resolve(url);
      tag.onerror = () => reject(url);
      document.head.appendChild(tag);
    }
  });
}

(async () => {
  // If you are concerned about display issues on mobile devices, you can use screen.width to determine whether to load
  // 如果担心手机上显示效果不佳，可以根据屏幕宽度来判断是否加载
  // if (screen.width < 768) return;

  // Decorative animation: honour prefers-reduced-motion like every other
  // effect on this page.
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  // Avoid cross-origin issues with image resources
  // 避免图片资源跨域问题
  const OriginalImage = window.Image;
  window.Image = function(...args) {
    const img = new OriginalImage(...args);
    img.crossOrigin = "anonymous";
    return img;
  };
  window.Image.prototype = OriginalImage.prototype;

  // Dock bottom-right: the bottom-left corner is this page's sidebar rail.
  // z-index: the widget defaults to 1, which the frontend's full-screen
  // canvas wrapper paints over — the model loads fine but stays invisible
  // (the tips bubble has its own stacking, which is why it showed anyway).
  const style = document.createElement('style');
  style.textContent =
    '#waifu{left:auto !important;right:14px !important;bottom:0 !important;z-index:99990 !important}';
  document.head.appendChild(style);

  // Load waifu.css and waifu-tips.js
  // 加载 waifu.css 和 waifu-tips.js
  await Promise.all([
    loadExternalResource(live2d_path + 'waifu.css', 'css'),
    loadExternalResource(live2d_path + 'waifu-tips.js', 'js')
  ]);
  // For detailed usage of configuration options, see README.en.md
  // 配置选项的具体用法见 README.md
  // The model library is SELF-HOSTED under ./live2d_models/ (Potion-Maker/Pio
  // from fghrsh/live2d_api): the default jsdelivr CDN 302s model textures to
  // raw.githubusercontent.com, which is unreachable from mainland networks
  // and fails the CORS-checked Image load, leaving the widget model-less
  // (hitTest crashes on every mousemove). Absolute URL resolved at runtime so
  // subpath deployments work.
  initWidget({
    waifuPath: live2d_path + 'waifu-tips.json',
    cdnPath: new URL('live2d_models/', location.href).href,
    cubism2Path: live2d_path + 'live2d.min.js',
    cubism5Path: 'https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js',
    tools: ['hitokoto', 'asteroids', 'switch-model', 'switch-texture', 'photo', 'info', 'quit'],
    logLevel: 'warn',
    drag: false,
  });
  console.log('[ComfyDL.Home] Live2D widget requested');
})();

console.log(`\n%cLive2D%cWidget%c\n`, 'padding: 8px; background: #cd3e45; font-weight: bold; font-size: large; color: white;', 'padding: 8px; background: #ff5450; font-size: large; color: #eee;', '');
