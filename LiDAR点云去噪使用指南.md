# LiDAR 点云去噪工具使用指南

本工具包含三个脚本，用于清理 LiDAR（`.las` / `.laz`）点云中的两类常见噪点：

| 脚本 | 作用 |
|---|---|
| `comet.py` | 剔除"彗星尾"伪影（Comet Tail，激光多次反射/混叠造成的拖尾噪点） |
| `scene_context_noise.py` | 对每个点打"噪声概率"分数（基于局部密度 + 几何形态） |
| `comet_noise_pipeline.py` | 一键串联以上两步：先去彗星尾，再做噪声打分 |

---

## 0. 环境准备

```bash
pip install laspy numpy scipy --break-system-packages
```

> 如果你的 `.las` 文件用的是 LAZ 压缩格式，还需要装一个后端：
> ```bash
> pip install "laspy[lazrs]" --break-system-packages
> ```

三个脚本放在同一个文件夹下即可（`comet_noise_pipeline.py` 需要 `import comet` 和 `import scene_context_noise`，所以三者必须在同一目录，或者在你的 `PYTHONPATH` 里）。

---

## 1. `comet.py`：去除彗星尾噪点

### 是什么

LiDAR 扫描时，如果激光打在边缘、半透明物体或快速移动的目标上，会产生沿着射线方向"拖尾"的虚假点云，俗称"彗星尾"。这个脚本的原理：

1. 把每个点投影到单位球面上，沿"角度方向"找邻居（即近似同一条激光射线方向上的点）；
2. 检查这些方向相近的点在"径向距离"上是否聚集在一段范围内（`min_radial_thresh` ~ `radial_thresh` 之间）——这正是拖尾的典型形态；
3. 再用空间邻域做一次"二次确认"：如果某点周围大部分邻居都已经被标记为彗星尾，它也会被顺带标记（去除孤立误判、把整条尾巴标全）。

### 命令行用法

```bash
python comet.py 输入文件.las 输出文件.las
```

带自定义参数：

```bash
python comet.py 输入文件.las 输出文件.las ^
    --angular-thresh 0.003 ^
    --min-radial-thresh 0.02 ^
    --radial-thresh 0.25 ^
    --max-neighbors 100 ^
    --min-neighbors 20 ^
    --batch-size 50000
```

> Windows 下用 `^` 换行，Linux/Mac 下用 `\`。

### 参数说明

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--angular-thresh` | 0.003（弧度） | 判定"同一条射线方向"的角度容差，越大越容易把更多点视为同方向 |
| `--min-radial-thresh` | 0.02 | 径向距离小于此值的邻居不计入"尾部"（排除离目标点太近、属于正常表面厚度的点） |
| `--radial-thresh` | 0.25 | 径向距离大于此值的邻居不计入"尾部"（拖尾的最大延伸范围） |
| `--max-neighbors` | 100 | 每次查询最多取多少个候选邻居 |
| `--min-neighbors` | 20 | 落在"尾部区间"内的邻居数达到这个数量，才判定该点是彗星尾 |
| `--batch-size` | 50000 | 每批处理的点数，越大越快但占用内存越多 |

**输出**：默认是过滤后的 `.las`（直接删除被判定为彗星尾的点，不保留被删点）。

### 调参建议

- 噪点没删干净 → 适当调大 `--angular-thresh`，或调小 `--min-neighbors`；
- 误删了正常点（比如细长的电线、栏杆被当成尾巴） → 调小 `--angular-thresh`，或调大 `--min-neighbors`，或缩小 `--radial-thresh` 的范围。

---

## 2. `scene_context_noise.py`：场景噪声打分

### 是什么

这个脚本不直接删点，而是给**每一个点**算一个 0~1 的"噪声概率" `noise_prob`，综合两类信号：

1. **密度稀疏度**：点到第 K 个邻居的距离，按"离扫描原点的远近"分 20 个区间分别归一化（避免远处天然稀疏的点被误判为噪声）。距离越远（局部越稀疏），分数越高。
2. **局部几何形态**（对邻域做 PCA 特征值分解）：
   - **线性度（linearity）**：邻域呈细线状（典型拖尾/孤立线）→ 高 → 偏噪声；
   - **低曲率（low curvature）**：邻域既不成面也不成线、杂乱分布 → 高 → 偏噪声。

最终：

```
noise_prob = 0.50 × 密度信号 + 0.35 × 线性度 + 0.15 × 低曲率
```

### 命令行用法

```bash
python scene_context_noise.py 输入文件.las 输出文件.las [阈值]
```

例如：

```bash
python scene_context_noise.py input.las output.las 0.5
```

- 不传阈值：只输出 `noise_prob`，不生成二值标签；
- 传阈值（0~1 之间，默认 0.5）：额外生成一个 `noise_label` 字段（1=噪声，0=干净），方便后续直接用这个字段筛选。

### 输出字段（写入到输出 `.las` 的 Extra Bytes）

| 字段名 | 含义 |
|---|---|
| `noise_prob` | 综合噪声概率，0~1，越大越可能是噪声 |
| `density_signal` | 单独的密度稀疏度信号 |
| `linearity` | 单独的线性度信号 |
| `low_curvature` | 单独的低曲率信号 |
| `noise_label` | （仅当传入阈值时）二值标签，1=噪声 |

> 注意：这个脚本**不会删除点**，只是打分、写标签。如果想真正过滤，需要你自己根据 `noise_prob` 或 `noise_label` 再筛一遍（见下方"进阶用法"）。

### 一个小细节：自动修复重复字段

如果一个文件被脚本处理了两次（比如不小心跑了两遍写入了同名的 Extra Bytes 字段），直接用 `laspy.read()` 会报错。这个脚本内置了 `_read_las_safe()`，会自动检测并修复这种"重复字段"问题，正常使用不需要关心这一点。

---

## 3. `comet_noise_pipeline.py`：一键跑完整流程

### 是什么

把"先去彗星尾、再打噪声分"这两步串起来，自动执行：

```
原始点云 → [comet.py 去彗星尾] → 中间文件 → [scene_context_noise.py 打分] → 最终文件
```

### 命令行用法

```bash
python comet_noise_pipeline.py 输入文件.las 输出文件.las
```

可选参数：

```bash
python comet_noise_pipeline.py 输入文件.las 输出文件.las ^
    --first-comet-output 中间文件.las ^
    --noise-threshold 0.5
```

| 参数 | 含义 |
|---|---|
| `input_file` | 原始点云文件（位置参数，可省略，省略则用脚本内写死的默认路径） |
| `output_file` | 最终输出文件（位置参数，同上） |
| `--first-comet-output` | 第一阶段（去彗星尾后）的中间文件保存路径 |
| `--noise-threshold` | 第二阶段噪声打分的阈值，传给 `scene_context_noise.run()` |

> ⚠️ **注意**：脚本里写了几个默认路径（`DEFAULT_INPUT_FILE` / `DEFAULT_FIRST_COMET_OUTPUT_FILE` / `DEFAULT_OUTPUT_FILE`），是原作者电脑上的本机路径（`D:\...` 和 `C:\Users\njzy1\...`）。**这些路径在你的电脑上一定不存在**，所以**必须**在命令行里显式传入你自己的输入/输出路径，否则会找不到文件直接报错退出。

### 推荐用法（显式传参，避免踩坑）

```bash
python comet_noise_pipeline.py "你的输入.las" "你的最终输出.las" --first-comet-output "中间结果.las"
```

第一阶段去彗星尾的中间结果也会被保留在磁盘上（不会自动删除），方便你检查两个阶段分别起了什么作用。

---

## 5. 进阶：用 `noise_prob` 自己做过滤

如果你想根据 `scene_context_noise.py` 算出来的概率，自己写一段过滤脚本（删除噪点而不是只打标签），可以参考：

```python
import laspy
import numpy as np

las = laspy.read("output.las")
keep = las.noise_prob < 0.5   # 保留噪声概率小于阈值的点
out = laspy.LasData(las.header.copy())
out.points = las.points[keep].copy()
out.write("output_filtered.las")
print(f"保留 {keep.sum():,} / {len(keep):,} 个点")
```

---

## 6. 常见问题

。

**Q: `scene_context_noise.py` 报错 "No extra-bytes VLR found"？**
A: 说明文件本身没有重复字段问题，但程序错误地走进了"修复重复字段"的分支——通常意味着这是别的原因导致的 `ValueError`（比如文件本身损坏）。检查报错信息里 `_read_las_safe` 之前的原始异常文本，能看到具体出错原因。

**Q: 想看看到底删了哪些点，而不是直接删掉？**
A: `comet.py` 里还有一个 `save_classified_las()` 函数（未在命令行里暴露），它会把分类结果写成一个 `comet_class` 字段（0=正常，1=尾部）而不是直接删点，方便你在点云软件（如 CloudCompare）里按字段着色查看。如果需要，可以照着 `save_filtered_las` 的调用方式，自己加一行调用它。
