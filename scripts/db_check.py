"""测试环境数据库只读自检。

只执行 SELECT / SHOW 这类只读语句，不会写任何数据。

用法：
    python scripts/db_check.py                  # 连通性 + 库信息
    python scripts/db_check.py --tables         # 列出所有表及行数
    python scripts/db_check.py --table t_name   # 看某张表的字段结构
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # Windows 控制台默认 cp936，中文输出会乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:  # pragma: no cover
    pass

from src.rag.config import db_settings  # noqa: E402


def connect(settings: dict):
    import pymysql

    return pymysql.connect(
        connect_timeout=8,
        read_timeout=30,
        autocommit=True,
        **settings,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="测试环境数据库只读自检")
    parser.add_argument("--tables", action="store_true", help="列出所有表及行数")
    parser.add_argument("--table", default=None, help="查看指定表的字段结构")
    args = parser.parse_args()

    settings = db_settings()
    if not settings["host"]:
        print("未配置 DB_HOST（.env 里填），跳过数据库自检。")
        return 0

    print(f"连接 {settings['host']}:{settings['port']}  user={settings['user']}  db={settings['database']}")
    print("（本脚本只发只读查询）\n")

    try:
        connection = connect(settings)
    except Exception as exc:  # noqa: BLE001 - 自检脚本，错误要原样告诉人
        print(f"连接失败：{exc}")
        print("\n排查方向：")
        print("  1. 本机到该端口的网络/安全组是否放行")
        print("  2. 账号是否允许从当前 IP 登录（MySQL 的 user@host 授权）")
        print("  3. .env 里的 DB_PASSWORD 是否需要 URL 编码之外的原始字符")
        return 1

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            print(f"MySQL 版本 : {cursor.fetchone()[0]}")

            cursor.execute("SELECT DATABASE()")
            print(f"当前库     : {cursor.fetchone()[0]}")

            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = %s",
                (settings["database"],),
            )
            table_count = cursor.fetchone()[0]
            print(f"表数量     : {table_count}")

            if args.tables and table_count:
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = %s ORDER BY table_name",
                    (settings["database"],),
                )
                names = [row[0] for row in cursor.fetchall()]
                print("\n表清单：")
                for name in names:
                    # 表名来自 information_schema，这里仍然做一次白名单校验再拼进 SQL
                    if not name.replace("_", "").isalnum():
                        print(f"  {name:<40} 跳过（表名含特殊字符）")
                        continue
                    cursor.execute(f"SELECT COUNT(*) FROM `{name}`")  # noqa: S608 - 表名已校验
                    print(f"  {name:<40} {cursor.fetchone()[0]:>8} 行")

            if args.table:
                cursor.execute(
                    "SELECT column_name, column_type, is_nullable "
                    "FROM information_schema.columns WHERE table_schema = %s AND table_name = %s "
                    "ORDER BY ordinal_position",
                    (settings["database"], args.table),
                )
                columns = cursor.fetchall()
                if not columns:
                    print(f"\n表 {args.table} 不存在")
                else:
                    print(f"\n表 {args.table} 结构：")
                    for name, column_type, nullable in columns:
                        print(f"  {name:<30} {column_type:<20} NULL={nullable}")
    finally:
        connection.close()

    print("\n只读自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
