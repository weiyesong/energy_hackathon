# Satellite Irradiance Forecast Container

这个目录提供一个隔离的 Docker 环境，用于卫星图像处理、太阳辐照度计算和电网发电预测建模。

## 包含能力

- 卫星/栅格数据：`satpy`, `pyresample`, `rasterio`, `rioxarray`, `xarray`, `netcdf4`, `h5py`, `zarr`
- 地理处理：`gdal`, `pyproj`, `geopandas`, `shapely`, `cartopy`
- 辐照度/PV：`pvlib-python`
- 预测建模：`scikit-learn`, `lightgbm`, `statsmodels`
- 数据访问：`pystac-client`, `odc-stac`, `planetary-computer`, `s3fs`
- 分析开发：`jupyterlab`, `matplotlib`, `plotly`, `pytest`, `ruff`

## 目录约定

```text
data/raw/        原始卫星图像、天气数据或电站历史发电量，默认只读挂载
data/processed/  中间结果，例如云掩膜、GHI/DNI/DHI 栅格、站点级特征
data/output/     预测结果、模型文件、评估图表
src/             项目代码
notebooks/       实验 notebook
```

## 构建

```bash
docker compose -f compose.yml build
```

## 运行隔离环境

默认服务运行时禁用网络，适合处理已经放在 `data/raw/` 的数据：

```bash
docker compose -f compose.yml run --rm irradiance
```

执行验证脚本：

```bash
docker compose -f compose.yml run --rm irradiance python src/smoke_test.py
```

## 需要联网下载数据时

使用带网络的 profile，只建议用于下载公开卫星或天气数据。下载后再切回默认隔离服务运行计算。

```bash
docker compose -f compose.yml --profile network run --rm irradiance-net
```

## JupyterLab

如果需要 notebook，请临时使用联网服务并暴露端口：

```bash
docker compose -f compose.yml --profile network run --rm -p 8888:8888 irradiance-net \
  jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=energy
```

打开 `http://localhost:8888`，token 是 `energy`。

## 建议的计算流程

1. 读取卫星影像或云产品，重投影到电站/电网区域。
2. 用云量、太阳高度角、地表反照率等特征估计 GHI/DNI/DHI。
3. 用 `pvlib` 将辐照度转换为组件平面辐照度和理论 PV 输出。
4. 合并历史发电量、天气预报和负荷/限电信息，训练短期预测模型。
5. 输出区域或站点级发电预测到 `data/output/`。
