# 使用 OpenFaaS 的经典 watchdog 作为基础镜像
FROM openfaas/classic-watchdog:0.18.1 as watchdog
# 设置工作目录
WORKDIR /app
# 复制项目文件
COPY ../src /app/
COPY ../depreciation_model.h5 /app/
COPY ../scaler.pkl /app/
COPY ../encoder.pkl /app/
COPY ../requirements.txt /app/

# 安装 Python 环境和依赖
FROM python:3.8-slim
# 将 watchdog 复制到新镜像中
COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog

# 设置工作目录
WORKDIR /app
# 安装 Python 依赖
RUN pip install -r requirements.txt
# 设置环境变量，让 OpenFaaS 使用 fwatchdog 来处理请求
ENV fprocess="python3 /app/handler.py"
ENV mode="http"
ENV write_debug="true"
ENV prefix_logs="false"
# 给 watchdog 设置可执行权限
RUN chmod +x /usr/bin/fwatchdog
# 监听的默认端口
EXPOSE 8080
# 设置启动命令
CMD ["fwatchdog"]

