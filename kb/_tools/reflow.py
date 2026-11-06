# -*- coding: utf-8 -*-
"""把 md 重排成"一段一行"：段内不再硬换行。

保留原样：空行、标题(#)、表格(|)、公式($$ 块)、代码块(```)、分隔线(---)、
         页面头部的 状态/可信度/最后核实/适用 四行、引用块里的列表项。
合并：正文段落、列表项的续行、引用块的续行。
"""
import io
import re
import sys

CJK = re.compile(r'[\u3000-\u9fff\uff00-\uffef]')
META = re.compile(r'^(状态|可信度|最后核实|适用):')
ITEM = re.compile(r'^(\s*)([-*]|\d+\.)\s+')


def join_lines(a, b):
    if not a:
        return b
    if not b:
        return a
    if CJK.match(a[-1]) and CJK.match(b[0]):
        return a + b
    return a + ' ' + b


def special_plain(line):
    s = line.strip()
    return s == '' or s.startswith('#') or s.startswith('|') or s.startswith('```') or s == '---'


def reflow(text):
    lines = text.split('\n')
    out, i = [], 0
    while i < len(lines):
        line, s = lines[i], lines[i].strip()

        # 页面头四行：各自独立
        if META.match(s):
            out.append(s)
            i += 1
            continue

        if special_plain(line) or s.startswith('$$'):
            # $$ 块整体原样
            if s.startswith('$$'):
                while i < len(lines):
                    out.append(lines[i].rstrip())
                    done = lines[i].strip().endswith('$$') and i > 0 and lines[i].strip() != '$$'
                    i += 1
                    if done:
                        break
                    if lines[i - 1].strip() == '$$':
                        break
            else:
                out.append(line.rstrip())
                i += 1
            continue

        # 引用块：逐行判断，列表项/空行不合并
        if s.startswith('>'):
            body = s[1:].strip()
            if body == '' or ITEM.match(body):
                out.append(s)
                i += 1
                continue
            buf = s
            i += 1
            while i < len(lines):
                t = lines[i].strip()
                if not t.startswith('>'):
                    break
                inner = t[1:].strip()
                if inner == '' or ITEM.match(inner):
                    break
                buf = join_lines(buf, inner)
                i += 1
            out.append(buf)
            continue

        # 列表项
        m = ITEM.match(line)
        if m:
            buf = line.rstrip()
            i += 1
            while i < len(lines):
                t = lines[i]
                if t.strip() == '' or special_plain(t) or t.strip().startswith('>') or ITEM.match(t):
                    break
                if t.startswith('  ') or t.startswith('\t'):
                    buf = join_lines(buf, t.strip())
                    i += 1
                else:
                    break
            out.append(buf)
            continue

        # 普通段落
        buf = s
        i += 1
        while i < len(lines):
            t = lines[i]
            if t.strip() == '' or special_plain(t) or t.strip().startswith('>') or ITEM.match(t) \
                    or t.strip().startswith('$$') or META.match(t.strip()):
                break
            buf = join_lines(buf, t.strip())
            i += 1
        out.append(buf)

    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    for p in sys.argv[1:]:
        t = io.open(p, encoding='utf-8').read()
        t2 = reflow(t)
        io.open(p, 'w', encoding='utf-8').write(t2)
        print(f'{p}: {t.count(chr(10))} -> {t2.count(chr(10))} 行')
