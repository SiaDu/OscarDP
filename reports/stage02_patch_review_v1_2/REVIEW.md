# Stage02 Patch Quality Review v1.2

**仅审核与修订提案；没有批准或应用任何修改。** APPROVE 表示该条审核建议可接受，不代表已经取得最终应用授权。

原有 70 条文本提案：{'APPROVE': 63, 'REVISE': 0, 'REJECT': 0, 'NEEDS_CONTEXT_REVIEW': 7}。
原有 62 条伴随对白：{'APPROVE': 49, 'REVISE': 6, 'REJECT': 0, 'NEEDS_CONTEXT_REVIEW': 7}。

46 个原始独立 Cue 均已查看相邻字幕与前后 Shot。确定变更涉及 **54 个独立 Shot**；另有 7 个 Shot 的原始提案因 4 个独立语义 Case 被扣留。原问题涉及 61 个 Shot。
62 条原伴随条目的 speaker 均保留 null，其中安全纳入子集的 55 条也保留 null。5 条带 MAN/WOMAN 源标签，但这些标签不足以映射为正式角色名。

## 已确认的修订

- Wicked cue 615：`Oh. -` 应为 `Oh.`。只移除紧接下一歌词说话轮次的分隔符，不全局删除连字符。
- Song Sung Blue cues 924、930、931、933：移除正文中的 MAN 2 / MAN 4 / MAN 5 / WOMAN SDH 标签；保留来源证据与 null speaker。930 跨两 Shot，处理一致。
- 其余可确认对白保持源词语、大小写、标点和索引。新增对白按原 Cue 顺序插入，所有既有 Dialogue 元素及相对顺序不变。不同 Cue 中重复的 Yes/Come on 不被误删。
- A Complete Unknown 的 8 条对应对白已经存在且索引正确，不重复添加。既有说话者、对白和 narrative 均保持不变。
- 另发现 6 处遗漏（5 个独立 Cue）：歌词中残留 [all] / CLAIRE 标签、已有独立音效或 singing-along 描述。已独立列为 additional_field_cleanups，不冒充原 70 条；已有音效保留且不重复添加。

## 文件级方案与内存验证

| Movie | 未改动 v1.1 Validator 结果 | 可提交最终批准 | 原因 |
| --- | --- | --- | --- |
| tt11563598 A Complete Unknown | PASS | 可以，仍待用户授权 | 审核范围内的精确修订通过自动检查 |
| tt1262426 Wicked | FAIL | 尚不可以 | Unresolved text-function decisions excluded from this subset; Unchanged v1.1 rejects some correctly cleaned boundary/SDH text; do not reinstate metadata to manufacture PASS |
| tt30343021 Song Sung Blue | FAIL | 尚不可以 | Unresolved text-function decisions excluded from this subset; Unchanged v1.1 rejects some correctly cleaned boundary/SDH text; do not reinstate metadata to manufacture PASS |
| tt31193180 Sinners | FAIL | 尚不可以 | Unresolved text-function decisions excluded from this subset |

**没有为获取 PASS 放宽或修改 Validator。** 正确清理分隔符/SDH 标签后，当前 v1.1 的完整片段比较仍会错误拒绝这 6 处规范化；详见 validation_preview.json 的 validator_support_gaps。独立来源审查已验证它们没有改变实际词语，但不能把此结果伪称为原 Validator PASS。

3 份未完成方案的顶层 json_patch 为 null，仅保留 reviewed_subset 的内存回放操作。被扣留条目不作修改；安全子集通过来源和结构检查，不等于整份文件已完成语义审核。每份方案均有原始 SHA-256、精确数组 Before/After 和 test guards，applied / final_application_authorized 均为 false。

## 真正待语义复核的 4 个 Case

- Wicked 1299：Hmm. 是口头应答、哼唱还是非语言声音？3 个 Shot。
- Song Sung Blue 2217、2220：James Brown 致敬中的 Yeah 是歌唱回应、普通应答还是欢呼？分别 2 / 1 个 Shot。
- Sinners 1217：Whoo 是对白感叹、非语言欢呼还是音乐表演的一部分？1 个 Shot。

已阅读相关剧本连续片段，但它们没有记录这些具体成片发声，不能替代原始 SRT 或音视频。unresolved_semantic_cases.json 已附独立上下文包，明确问题与需要的补充依据；没有把 null speaker 本身扩大为语义 Case。

## 证据、测试与本地文件

- patch_review.csv 恰为 70 行；companion_dialogue_review.csv 恰为 62 行。完整原文、索引、审核理由、规范化说明、重复检查和说话者证据可逐行追溯。
- 原始 Stage01 / Stage02 / SRT、参考 PDF、固定 Spec / Prompt 和审核输入 SHA-256 在生成前后复核一致。修订副本只存在内存；没有写入正式目录或输出候选 Stage02 JSON。
- validation_preview.json 保留实际执行的 Schema / Coverage / Temporal / SRT / Text / Classification 结果、错误和警告，以及独立来源检查。自动通过、语义已审和仍未确认分别记录。
- 自动化测试：仓库原有 133 项通过；本次临时审核回归 20 项通过，覆盖分隔符、SDH 标签、真实连字符/负号、test guard、未决排除、合法跨 Cue 同词重复、补充遗漏与实际 Validator 结果。未修改项目代码或原有测试。
- 受影响 Shot 中另有既有 Dialogue 的 SDH 标签，已列为 original_context_entries_left_unchanged；这些不属于原 62 条新增对白，未借本次审核覆盖既有条目。
- 工作期间，仓库新提交 02164f3 删除了旧 v1.1 报告目录。本次从现存提交 4c78745 只读恢复参考到 /tmp，未恢复工作区目录。该历史提交仍包含字幕证据和衍生 SRT；未执行 git add、commit、push 或历史改写。
- 本目录含字幕证据，仅供本地审核，不应推送 Public GitHub。当前 .gitignore 未覆盖新目录；文件未暂存或提交，未擅自修改你的忽略配置。

**停止在待批准阶段。** 只有 A Complete Unknown 的本次方案可提交最终批准；其余三份需要先处理明确的上下文问题与 Validator 支持缺口。
