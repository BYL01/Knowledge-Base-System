# 离线部署手册

目标环境：**Linux 服务器、没有 Docker、没有 Jenkins，只要把评测跑批放上去**。

实测记录（2026-09-30，Alibaba Cloud Linux 3 云主机，主机地址已脱敏）：

- 服务器**能上外网**（pypi 200、阿里云镜像 301、api.deepseek.com 401），原以为没外网，实际不用预热判官缓存；
- 系统自带 `python3` 只有 3.6.8，不满足代码要求的 3.9+，先装运行时：

  ```bash
  yum install -y python3.11 python3.11-pip python3.11-setuptools
  ```

- 安装时用环境变量指定解释器（否则会拿到 3.6）：

  ```bash
  cd /tmp/rag-eval-offline && ./install.sh --with-timer          # 注意：若用顶层 wrapper
  # 或直接跑内层：
  cd /tmp/rag-eval-offline/app && PYTHON_BIN=/usr/bin/python3.11 ./deploy/install.sh --with-timer
  ```

## 一、在本地打包（这台 Windows 机器有网）

```powershell
py -3.11 deploy/build_bundle.py
```

产物：`dist\rag-eval-offline-<日期>.tar.gz`（约 19 MB，其中 18 MB 是 Linux 版 numpy 的 wheel）。

包内结构：

```
rag-eval-offline/
  install.sh                     顶层入口
  rag-eval-offline/app/          项目代码
  rag-eval-offline/app/wheels/   Linux 依赖 wheel（离线安装用，无需外网）
```

如果服务器上的 Python 不是 3.11，打包时指定版本，例如 Python 3.10：

```powershell
py -3.11 deploy/build_bundle.py --python-version 310
```

## 二、上传并安装

```bash
scp dist/rag-eval-offline-*.tar.gz user@server:/tmp/

# 服务器上
cd /tmp
tar xzf rag-eval-offline-*.tar.gz
cd rag-eval-offline
./install.sh --with-timer
```

install.sh 做七件事：检查 Python → 拷代码到 `/opt/rag-eval` → 用本地 wheel 建虚拟环境 → 生成 `.env` → 建索引 → 跑冒烟测试 → 装 systemd 每日定时任务。

可以用环境变量改路径：

```bash
INSTALL_DIR=/srv/rag-eval LOG_DIR=/srv/rag-eval/logs ./install.sh --with-timer
```

## 三、跑批与门禁

```bash
cd /opt/rag-eval
./deploy/run_eval.sh          # 建索引 -> 跑 golden 集 -> 质量门禁
echo $?                       # 0 通过；非 0 表示门禁不通过
```

定时任务每天 02:30 跑一次（带 5 分钟随机抖动，关机错过的会自动补跑）：

```bash
sudo systemctl list-timers rag-eval.timer
sudo systemctl start rag-eval.service     # 手动触发一次
sudo journalctl -u rag-eval.service -n 80 # 看日志
cat /var/log/rag-eval/eval-$(date +%Y%m%d).log
```

跑批产物在 `outputs/runs/`：`<run_id>.jsonl` 是每条问答的完整中间产物，`<run_id>.summary.json` 是指标汇总。

## 四、无外网时判官怎么办（重点）

DeepSeek 判官要发 HTTP 请求。服务器没外网时，有三种做法，按推荐度排序：

**做法 A：先在本地预热缓存，再连缓存一起打包（推荐，完全离线）**

1. 在这台有网的机器上，把 `.env` 里的 `DEEPSEEK_API_KEY` 填好；
2. 把 `configs/baseline.yaml` 里 `answerability.mode` 改成 `llm`；
3. 跑一次 `py -3.11 scripts/run_batch.py`，判官结果会落到 `data/judge_cache/`（按内容哈希缓存，重复跑不重复花钱）；
4. 再执行 `py -3.11 deploy/build_bundle.py --with-judge-cache`，脚本会把 `data/judge_cache/` 一起打进包里；
5. 服务器上把 `judge.cache_only` 设为 `true`——判官只读缓存、不发请求；缓存没命中的题按 `answerability.on_error` 兜底。

**做法 B：服务器直连 DeepSeek**

如果服务器虽然上不了外网、但能访问 `api.deepseek.com`（国内域名，不少内网也通），直接设：

```yaml
answerability:
  mode: llm
judge:
  cache_only: false
```

先跑 `./deploy/check_env.sh` 确认连通性，它会真的发一次请求测。

**做法 C：先不启用判官（当前默认）**

`answerability.mode: off` 时判官完全不参与，跑批纯本地、确定性、秒级完成。可以先这样把跑批和定时任务跑顺，判官后面再加。

## 五、离线包的依赖清单

包里的 wheel 覆盖：numpy（向量计算）、PyYAML（配置）、requests（调 DeepSeek）、pytest（冒烟测试）及其依赖。
装的是 Linux x86_64 + CPython 3.11 版本；换平台或 Python 版本时用 `build_bundle.py` 的参数重新打包。

## 六、常见问题

**`install.sh: Permission denied`** —— 压缩包经过 Windows 中转可能丢执行位，`chmod +x install.sh deploy/*.sh` 一下。

**没有 systemd（容器或老系统）** —— 安装脚本会提示改用 cron，加这一行即可：

```
30 2 * * * cd /opt/rag-eval && INSTALL_DIR=/opt/rag-eval ./deploy/run_eval.sh
```

**`python3 -m venv` 失败** —— 多半是缺 `python3-venv` 包。不能联网装也没关系，脚本会自动退回 `pip install --user`。

**门禁返回非 0** —— 这是预期行为，说明指标没达到 `configs/baseline.yaml` 里 `gate` 的要求。看 `outputs/runs/<run_id>.summary.json` 是哪一项不达标，别去调松阈值。

**想回滚** —— `/opt/rag-eval` 是纯代码目录，保留上一版 tar 包重新解压覆盖即可；索引可随时重建，产物都在 `outputs/`。
