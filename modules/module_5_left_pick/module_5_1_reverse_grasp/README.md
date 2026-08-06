# Module 5-1: Reverse Grasp

`main.py` 是保留的旧一体化入口，供旧流程单独运行，行为不变。

模块化总控使用 `execute_prediction.py`：它只接受 Module 4-1 的 prediction JSON，
执行左臂反抓预备位和真实抓取；不控制云台、不采集图像、不调用 SAM2/ICP。

`implementation/` 和 `main.py` 的一体化感知入口保留，供旧总控和独立调试使用；
模块化调用链不会经过它们。

```bash
/home/dainanjun/Public/data_collector/.venv/bin/python execute_prediction.py \
  --execute --prediction-file /tmp/reverse_grasp_prediction.json
```
