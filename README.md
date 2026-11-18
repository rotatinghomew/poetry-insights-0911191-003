# 唐诗工坊满额下的真实需求与增场评估

本项目保存诗词工坊报名和场次安排的合成资料，包含申请偏好、预约事件及场次资源。资料使用 UTF-8 CSV，空字段表示记录未提供。

## 数据

- `data/requests.csv`：申请人及时段偏好顺序。
- `data/booking_events.csv`：确认、候补、回应及撤销事件。
- `data/sessions.csv`：工坊、讲师、场地和最低开课人数。

## 环境与验证

需要 Python 3.11 或更新版本；项目仅使用 Python 标准库。可用 `python3 -m unittest discover -s tests -v` 检查 CSV 结构与本地资料完整性。
