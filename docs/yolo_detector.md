# YOLODetector 说明

本文档说明 [`src/nero_workcell/core/yolo_detector.py`](../src/nero_workcell/core/yolo_detector.py) 中 YOLO 检测与深度反投影的核心原理，重点解释以下问题：

- 检测器输入输出的物理含义
- `_estimate_depth()` 为什么要用局部窗口和中位数
- 像素坐标为什么能恢复为相机坐标系中的三维点

## 1. 整体流程

`YOLODetector.detect_objects()` 的处理流程如下：

1. 对彩色图运行 YOLO，得到检测框 `bbox = [x1, y1, x2, y2]`
2. 计算检测框中心点 `center_x, center_y`
3. 在深度图中，以检测框中心为中心取一个局部窗口
4. 过滤掉无效深度值，取有效深度的中位数作为 `depth_value`
5. 结合相机内参，将 `(center_x, center_y, depth_value)` 反投影到相机坐标系
6. 输出 `TargetObject(frame="camera")`

最终得到的 `TargetObject.position = [X, Y, Z]` 表示目标在相机坐标系中的位置，单位为米。

## 2. 深度的物理含义

这里的深度 `depth_value` 指的是目标点在相机坐标系下，沿相机光轴方向的距离，也就是三维点的 `Z` 值。

它的物理含义是：

- 目标在相机前方多远

它不是：

- 目标点到相机光心的欧氏直线距离

如果相机坐标系中的三维点是 `(X, Y, Z)`，那么：

- `Z` 是深度
- `sqrt(X^2 + Y^2 + Z^2)` 才是空间中的直线距离

## 3. `_estimate_depth()` 的原理

对应代码逻辑：

```python
def _estimate_depth(self, depth, center_x, center_y):
    h, w = depth.shape
    radius = self.depth_window_radius
    region = depth[
        max(0, center_y - radius):min(h, center_y + radius),
        max(0, center_x - radius):min(w, center_x + radius),
    ]
    valid = region[region > 0]
    return float(np.median(valid)) if valid.size > 0 else 0.0
```

核心思想是：不要直接使用单个像素的深度，而是使用目标中心附近一小块区域的有效深度中位数。

这样做的原因：

- 单个像素容易受噪声、空洞、反光和边缘误差影响
- 深度图中 `0` 往往表示无效深度，不能直接参与计算
- 中位数比均值更抗异常值，鲁棒性更好

### 3.1 为什么要做边界裁剪

代码中的：

```python
max(0, center_y - radius):min(h, center_y + radius)
```

表示在图像行方向上取窗口时，要把起止位置限制在合法范围内。

原因是：

- 图像最小行索引是 `0`
- 图像最大行索引是 `h - 1`
- 如果目标靠近图像边缘，`center_y - radius` 可能小于 `0`
- 同理，`center_y + radius` 可能超过 `h`

列方向上的处理完全相同。

### 3.2 为什么只保留 `> 0` 的深度

代码中的：

```python
valid = region[region > 0]
```

表示只保留物理上有效的深度值。

原因是：

- `0` 在 RealSense 深度图中通常表示缺测或无效值
- 真实深度应该是正数
- 如果把 `0` 混进来，中位数会被拉低，导致目标位置明显错误

### 3.3 一个具体例子

假设：

- 检测框中心为 `(center_x, center_y) = (140, 170)`
- `depth_window_radius = 5`

那么会在深度图中取一个局部区域：

```python
depth[165:175, 135:145]
```

如果这个区域中的有效深度大多在 `0.62m` 左右，而少量像素是 `0` 或异常值，那么中位数可能是：

```python
depth_value = 0.62
```

这表示目标中心点在相机前方约 `0.62m`。

## 4. 像素坐标到相机坐标的反投影

对应代码：

```python
position = np.array(
    [
        (center_x - self.cx) * depth_value / self.fx,
        (center_y - self.cy) * depth_value / self.fy,
        depth_value,
    ],
    dtype=float,
)
```

这段代码使用的是针孔相机模型。

### 4.1 成像公式

三维点 `(X, Y, Z)` 投影到图像像素 `(u, v)` 时，有：

```python
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

其中：

- `fx, fy` 是焦距，单位是像素
- `cx, cy` 是主点，表示光轴在图像上的投影位置
- `u, v` 是图像中的像素坐标

### 4.2 反投影公式

如果现在已知：

- 像素坐标 `(u, v)`，也就是检测框中心 `(center_x, center_y)`
- 深度 `Z = depth_value`

就可以反推出相机坐标系中的三维点：

```python
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth_value
```

代码中的 `position = [X, Y, Z]` 就是这个公式的直接实现。

## 5. 反投影后的物理意义

如果得到：

```python
position = [0.05, 0.025, 0.50]
```

那么它的含义是：

- 目标在相机前方 `0.50m`
- 相对相机光轴中心，水平方向偏移 `0.05m`
- 相对相机光轴中心，垂直方向偏移 `0.025m`

也就是说，这个点不再是图像中的二维像素点，而是相机坐标系中的真实三维位置。

## 6. 一个完整数值例子

假设相机内参为：

```python
fx = 600
fy = 600
cx = 320
cy = 240
```

某次检测得到：

```python
center_x = 380
center_y = 270
depth_value = 0.50
```

代入反投影公式：

```python
X = (380 - 320) * 0.50 / 600 = 0.05
Y = (270 - 240) * 0.50 / 600 = 0.025
Z = 0.50
```

所以最终：

```python
position = [0.05, 0.025, 0.50]
```

物理上表示：

- 目标位于相机前方 `50cm`
- 同时相对主点向右偏 `5cm`
- 同时相对主点向下偏 `2.5cm`

## 7. 与 `ObjectFollower` 的关系

`YOLODetector` 输出的 `TargetObject` 位于相机坐标系中，即：

```python
frame = "camera"
```

在 [`src/nero_workcell/tasks/object_follower_3d.py`](../src/nero_workcell/tasks/object_follower_3d.py) 中，后续还会结合手眼标定矩阵和当前机械臂位姿，把这个点进一步变换到机械臂基坐标系中，供跟随控制使用。

可以概括为：

1. YOLODetector：像素坐标 + 深度 -> 相机坐标
2. ObjectFollower：相机坐标 -> 机械臂 base 坐标
