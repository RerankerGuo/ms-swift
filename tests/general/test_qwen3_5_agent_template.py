# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression tests for #10255.

With the Qwen3.5/3.6 swift backend, an assistant turn that carries both ``<think>...</think>`` reasoning
and ``tool_calls`` lost its reasoning whenever a tool message sat between the user prompt and that turn:
``_remove_history_thinking`` / ``_add_non_thinking_prefix`` took the last ``tool`` message as the round
boundary and treated the current agent round as history. The official ``chat_template.jinja`` (used by
vLLM / transformers at inference) only looks at the last ``user`` message, so the trained format differed
from the served one (the train/infer mismatch behind #9234).

``TestQwen3_5AddToolCallPrefix`` and ``TestGetLastUserRoundIncludeTool`` are CPU-only unit tests.
``TestQwen3_5EncodeMatchesJinja`` is the encode-level check: it needs the tokenizer (no weights are
downloaded) and compares ``template.encode`` with ``tokenizer.apply_chat_template`` token by token.
"""
import copy
import json
import os
import unittest
from functools import lru_cache

from swift.agent_template import agent_template_map
from swift.template.base import Template
from swift.template.utils import get_last_user_round


class TestQwen3_5AddToolCallPrefix(unittest.TestCase):
    """``_add_tool_call_prefix`` only adds the ``\\n\\n`` that the jinja inserts between the effective
    (post-``</think>``) assistant content and ``<tool_call>``. It must not prepend the preceding assistant
    content: that message stays in ``messages`` and is merged with the tool call afterwards, so prepending
    it here renders the reasoning twice.
    """

    def setUp(self):
        self.tpl = agent_template_map['qwen3_5']()
        self.tool_call_msg = {
            'role': 'tool_call',
            'content': json.dumps({
                'name': 'search',
                'arguments': {
                    'query': 'stock'
                },
            }),
        }
        self.tool_content = self.tpl._format_tool_calls([self.tool_call_msg])

    def test_post_think_text_adds_separator_only(self):
        pre = {'role': 'assistant', 'content': '<think>\nplan\n</think>\n\nSome preamble text.'}
        out = self.tpl._add_tool_call_prefix(self.tool_content, pre)
        self.assertEqual(out, '\n\n' + self.tool_content)

    def test_text_without_think_adds_separator_only(self):
        pre = {'role': 'assistant', 'content': 'Some preamble text.'}
        out = self.tpl._add_tool_call_prefix(self.tool_content, pre)
        self.assertEqual(out, '\n\n' + self.tool_content)

    def test_pure_thinking_adds_nothing(self):
        for content in ('<think>\nonly thinking\n</think>\n\n', '<think>\nonly thinking\n</think>'):
            pre = {'role': 'assistant', 'content': content}
            self.assertEqual(self.tpl._add_tool_call_prefix(self.tool_content, pre), self.tool_content)

    def test_no_pre_message_returns_tool_content_unchanged(self):
        self.assertEqual(self.tpl._add_tool_call_prefix(self.tool_content, None), self.tool_content)

    def test_non_assistant_pre_message_returns_tool_content_unchanged(self):
        pre = {'role': 'user', 'content': 'no preceding assistant here'}
        self.assertEqual(self.tpl._add_tool_call_prefix(self.tool_content, pre), self.tool_content)

    def test_empty_string_content_returns_tool_content_unchanged(self):
        pre = {'role': 'assistant', 'content': ''}
        self.assertEqual(self.tpl._add_tool_call_prefix(self.tool_content, pre), self.tool_content)


class TestGetLastUserRoundIncludeTool(unittest.TestCase):
    """`_remove_history_thinking` and `_add_non_thinking_prefix` in
    ``swift.template.base`` must use the last *user* message as the boundary
    for stripping historical reasoning (not the last user-or-tool), to match
    the official Qwen3.5/3.6 chat_template.jinja. The function accepts an
    ``include_tool`` keyword since #9923; #10255 fixes the two call sites.
    """

    def test_include_tool_false_ignores_tool_messages(self):
        messages = [
            {
                'role': 'user',
                'content': 'first user'
            },
            {
                'role': 'assistant',
                'content': 'first assistant'
            },
            {
                'role': 'tool',
                'content': 'tool result'
            },
            {
                'role': 'assistant',
                'content': 'second assistant'
            },
        ]
        self.assertEqual(get_last_user_round(messages, include_tool=False), 0)
        self.assertEqual(get_last_user_round(messages, include_tool=True), 2)

    def test_include_tool_false_matches_first_user(self):
        messages = [
            {
                'role': 'user',
                'content': 'only user'
            },
            {
                'role': 'assistant',
                'content': 'a1'
            },
            {
                'role': 'tool',
                'content': 't1'
            },
            {
                'role': 'assistant',
                'content': 'a2'
            },
            {
                'role': 'tool',
                'content': 't2'
            },
            {
                'role': 'assistant',
                'content': 'a3'
            },
        ]
        # The last user message is at index 0; everything from index 1
        # onward (including all tool messages and assistants after them)
        # belongs to the current round, so reasoning there must be
        # preserved by `_remove_history_thinking`.
        self.assertEqual(get_last_user_round(messages, include_tool=False), 0)

    def test_template_history_thinking_calls_use_include_tool_false(self):
        """Static guard so a future refactor of the two call sites cannot
        silently regress to the old (include_tool=True default) behaviour."""
        import inspect
        for name in ('_remove_history_thinking', '_add_non_thinking_prefix'):
            method = getattr(Template, name)
            source = inspect.getsource(method)
            self.assertIn(
                'get_last_user_round(messages, include_tool=False)',
                source,
                msg=(f'{name} must call '
                     '`get_last_user_round(messages, include_tool=False)` '
                     'to match the official jinja boundary; #10255'),
            )


# Only the tokenizer / config files are fetched (``load_model=False``), never the weights. Any Qwen3.5/3.6 checkpoint
# with the official chat template works; point the variable at a local directory to run offline.
MODEL_ID = os.getenv('QWEN3_5_TEST_MODEL') or 'Qwen/Qwen3.5-35B-A3B'

TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'search',
        'description': 'Search.',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string'
                }
            },
            'required': ['query']
        },
    },
}, {
    'type': 'function',
    'function': {
        'name': 'read',
        'description': 'Read a page.',
        'parameters': {
            'type': 'object',
            'properties': {
                'url': {
                    'type': 'string'
                }
            },
            'required': ['url']
        },
    },
}]


def _call(name, **arguments):
    return {'type': 'function', 'function': {'name': name, 'arguments': arguments}}


def _think(text):
    return f'<think>\n{text}\n</think>\n\n'


def _user(text):
    return {'role': 'user', 'content': text}


def _assistant(content, *tool_calls):
    message = {'role': 'assistant', 'content': content}
    if tool_calls:
        message['tool_calls'] = list(tool_calls)
    return message


def _tool(name, content):
    return {'role': 'tool', 'name': name, 'content': content}


# (messages, number of user turns). Every conversation ends with an assistant turn, like a training sample.
CASES = {
    # the repro from #10255: reasoning + tool_calls, then reasoning + answer
    'issue_repro': [
        _user('Check the stock and draft an order email if it is low.'),
        _assistant(_think('I need to check the stock first.'), _call('search', query='stock')),
        _tool('search', 'Stock: 3 left.'),
        _assistant(_think('3 is low, so an order is needed.') + 'Stock is low (3 left). Here is the draft.'),
    ],
    'two_tool_steps': [
        _user('q'),
        _assistant(_think('t1'), _call('search', query='a')),
        _tool('search', 'r1'),
        _assistant(_think('t2'), _call('read', url='u')),
        _tool('read', 'r2'),
        _assistant(_think('t3') + 'done'),
    ],
    'reasoning_and_preamble_before_call': [
        _user('q'),
        _assistant(_think('t1') + 'Let me search.', _call('search', query='a')),
        _tool('search', 'r1'),
        _assistant(_think('t2') + 'answer'),
    ],
    'parallel_calls': [
        _user('q'),
        _assistant(_think('t1'), _call('search', query='a'), _call('read', url='u')),
        _tool('search', 'r1'),
        _tool('read', 'r2'),
        _assistant(_think('t2') + 'answer'),
    ],
    'parallel_calls_with_preamble': [
        _user('q'),
        _assistant(_think('t1') + 'pre', _call('search', query='a'), _call('read', url='u')),
        _tool('search', 'r1'),
        _tool('read', 'r2'),
        _assistant(_think('t2') + 'answer'),
    ],
    'call_without_reasoning': [
        _user('q'),
        _assistant('', _call('search', query='a')),
        _tool('search', 'r1'),
        _assistant('answer'),
    ],
    'preamble_without_reasoning': [
        _user('q'),
        _assistant('Let me search.', _call('search', query='a')),
        _tool('search', 'r1'),
        _assistant('answer'),
    ],
    'empty_reasoning': [
        _user('q'),
        _assistant(_think(''), _call('search', query='a')),
        _tool('search', 'r1'),
        _assistant(_think('') + 'answer'),
    ],
    'system_prompt_and_plain_final_answer': [
        {
            'role': 'system',
            'content': 'You are a helpful agent.'
        },
        _user('q'),
        _assistant(_think('t1'), _call('search', query='a')),
        _tool('search', 'r1'),
        _assistant('answer without reasoning'),
    ],
    # the earlier user turn is history: the jinja drops its reasoning, the current turn keeps it
    'history_round_with_calls': [
        _user('q1'),
        _assistant(_think('h1'), _call('search', query='a')),
        _tool('search', 'r1'),
        _assistant(_think('h2') + 'a1'),
        _user('q2'),
        _assistant(_think('c1'), _call('read', url='u')),
        _tool('read', 'r2'),
        _assistant(_think('c2') + 'a2'),
    ],
    'history_round_parallel_calls_with_preamble': [
        _user('q1'),
        _assistant(_think('h1') + 'pre', _call('search', query='a'), _call('read', url='u')),
        _tool('search', 'r1'),
        _tool('read', 'r2'),
        _assistant(_think('h2') + 'a1'),
        _user('q2'),
        _assistant(_think('c1') + 'a2'),
    ],
    'history_call_without_reasoning': [
        _user('q1'),
        _assistant('', _call('search', query='a')),
        _tool('search', 'r1'),
        _assistant('a1'),
        _user('q2'),
        _assistant(_think('c1'), _call('read', url='u')),
        _tool('read', 'r2'),
        _assistant('a2'),
    ],
    'three_user_turns': [
        _user('q1'),
        _assistant(_think('h1'), _call('search', query='a')),
        _tool('search', 'r1'),
        _assistant('a1'),
        _user('q2'),
        _assistant(_think('h3'), _call('search', query='b')),
        _tool('search', 'r2'),
        _assistant(_think('h4') + 'a2'),
        _user('q3'),
        _assistant(_think('c1'), _call('search', query='c')),
        _tool('search', 'r3'),
        _assistant(_think('c2') + 'a3'),
    ],
}


@lru_cache(maxsize=1)
def _get_processor():
    from swift.model import get_processor
    return get_processor(MODEL_ID)


class TestQwen3_5EncodeMatchesJinja(unittest.TestCase):
    """``template.encode`` (swift backend, default arguments) must give the same token ids as the official
    chat template for assistant turns that carry reasoning and/or ``tool_calls``.

    The one deliberate difference is ``loss_scale='all'``: it keeps the reasoning of earlier user turns
    (``preserve_thinking`` defaults to True there, because every turn is trained on), while the jinja
    strips it. So with several user turns, ``loss_scale='all'`` is compared with ``preserve_thinking=False``.
    """

    def _encode(self, messages, *, loss_scale, **kwargs):
        from swift.template import get_template
        template = get_template(_get_processor(), template_type='qwen3_5', loss_scale=loss_scale, **kwargs)
        template.set_mode('train')
        return template.encode({'messages': copy.deepcopy(messages), 'tools': copy.deepcopy(TOOLS)})['input_ids']

    def _jinja(self, messages):
        processor = _get_processor()
        tokenizer = getattr(processor, 'tokenizer', processor)
        encoded = tokenizer.apply_chat_template(
            copy.deepcopy(messages), tools=copy.deepcopy(TOOLS), tokenize=True, return_dict=True)
        return list(encoded['input_ids'])

    def _assert_same(self, name, messages, **kwargs):
        template_ids = self._encode(messages, **kwargs)
        jinja_ids = self._jinja(messages)
        tokenizer = getattr(_get_processor(), 'tokenizer', _get_processor())
        self.assertEqual(
            template_ids,
            jinja_ids,
            msg=(f'{name} {kwargs}\n--- template.encode ---\n{tokenizer.decode(template_ids)}'
                 f'\n--- chat_template.jinja ---\n{tokenizer.decode(jinja_ids)}'))

    def test_last_round(self):
        for name, messages in CASES.items():
            with self.subTest(case=name):
                self._assert_same(name, messages, loss_scale='last_round')

    def test_all_single_user_turn(self):
        for name, messages in CASES.items():
            if sum(m['role'] == 'user' for m in messages) > 1:
                continue
            with self.subTest(case=name):
                self._assert_same(name, messages, loss_scale='all')

    def test_all_without_preserving_history_thinking(self):
        for name, messages in CASES.items():
            with self.subTest(case=name):
                self._assert_same(name, messages, loss_scale='all', preserve_thinking=False)

    def test_reasoning_with_tool_calls_is_rendered_once(self):
        """#10255 dropped it; the first attempt to fix it rendered it twice."""
        messages = CASES['issue_repro']
        tokenizer = getattr(_get_processor(), 'tokenizer', _get_processor())
        for loss_scale in ('last_round', 'all'):
            text = tokenizer.decode(self._encode(messages, loss_scale=loss_scale))
            with self.subTest(loss_scale=loss_scale):
                self.assertEqual(text.count('I need to check the stock first.'), 1)
                self.assertEqual(text.count('<think>\n\n</think>'), 0)


if __name__ == '__main__':
    unittest.main()
