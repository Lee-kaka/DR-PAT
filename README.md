# Introduction
Recommender systems (RSs), as a data-driven way, have been widely applied in various domains such as e-commerce and social media. However, RSs face long-term threats that require high attention. Attackers manipulate recommendation results by injecting malicious data for profit. These endless attacks seriously affect the accuracy and fairness of recommendations. In this paper, we propose SHARE (SHoot the ARrow at the targEt), a novel personalized adversarial training method driven by dynamic rewards to defend against these threats. First, we selectively eliminate low-value data to reduce the risk of poisoning samples contaminating the model by quantifying the contribution of samples to the robustness and fairness of the model. Second, we design a dynamic perturbation strategy based on user embedding norm to address the behavioral differences among different users. Finally, we introduce a multi-dimensional reward function to guide the model to dynamically balance defense effectiveness and recommendation quality in adversarial training. Extensive experiments with different basic recommendation models, different datasets, and different types of attacks, demonstrate the effectiveness of SHARE compared with existing competing baselines. Specifically, SHARE can improve the recommendation performance by an average of 19.92% in clean data scenarios and reduce the success rate of attacks by an average of 20.50%.
# Requirements
For our running environment see requirements.txt
# Usage
```python main.py```
<!-- 嵌入 HTML -->
<div>
  <pre><code id="code-block" class="language-python">
# 示例代码
print("Hello, World!")
</code></pre>
  <button id="copy-button" onclick="copyCode()">复制代码</button>
</div>

<script>
  // JavaScript 实现复制功能
  function copyCode() {
    // 获取代码块的内容
    const codeBlock = document.getElementById('code-block');
    const range = document.createRange();
    range.selectNode(codeBlock);
    window.getSelection().addRange(range);

    try {
      // 执行复制操作
      document.execCommand('copy');
    } catch (err) {
      console.error('复制失败', err);
    }

    // 清除选区
    window.getSelection().removeAllRanges();
  }
</script>
