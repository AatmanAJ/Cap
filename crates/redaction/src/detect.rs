use std::ops::Range;

use crate::SensitiveKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DetectorOptions {
    pub emails: bool,
    pub cards: bool,
    pub secrets: bool,
    pub phones: bool,
    pub faces: bool,
}

impl Default for DetectorOptions {
    fn default() -> Self {
        Self {
            emails: true,
            cards: true,
            secrets: true,
            phones: true,
            faces: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextMatch {
    pub kind: SensitiveKind,
    /// Byte range into the scanned line.
    pub range: Range<usize>,
    /// What identifies the item across frames: the normalized text.
    pub key: String,
}

/// Every sensitive span in one recognized line of text. Overlapping spans
/// are merged into one, keeping the strongest kind.
pub fn find_sensitive_text(line: &str, options: &DetectorOptions) -> Vec<TextMatch> {
    let chars = Chars::new(line);
    let mut found = Vec::new();
    if options.emails {
        find_emails(&chars, &mut found);
    }
    if options.cards {
        find_cards(&chars, &mut found);
    }
    if options.secrets {
        find_prefixed_keys(&chars, &mut found);
        find_jwts(&chars, &mut found);
        find_bearer_tokens(&chars, &mut found);
        find_telegram_tokens(&chars, &mut found);
        find_labeled_secrets(&chars, &mut found);
        find_high_entropy(&chars, &mut found);
        find_pem_body(&chars, &mut found);
    }
    if options.phones {
        find_phones(&chars, &mut found);
    }
    merge_overlaps(line, found)
}

/// [`find_sensitive_text`] for the lines of one frame, top to bottom, plus
/// what only shows across lines: every line inside a `-----BEGIN ... PRIVATE
/// KEY-----` block is key material.
pub fn find_sensitive_lines(lines: &[&str], options: &DetectorOptions) -> Vec<Vec<TextMatch>> {
    let mut in_private_key = false;
    lines
        .iter()
        .map(|line| {
            let mut found = find_sensitive_text(line, options);
            let upper = line.to_ascii_uppercase();
            let armour = upper.trim_start().starts_with(['-', '—', '–']);
            let begins = armour && upper.contains("BEGIN") && upper.contains("PRIVATE KEY");
            let ends = armour && upper.contains("END");
            if options.secrets && in_private_key && !ends && !begins {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    let start = line.len() - line.trim_start().len();
                    found = merge_overlaps(
                        line,
                        found
                            .into_iter()
                            .chain([TextMatch {
                                kind: SensitiveKind::Secret,
                                range: start..start + trimmed.len(),
                                key: trimmed.to_string(),
                            }])
                            .collect(),
                    );
                }
            }
            if begins {
                in_private_key = true;
            } else if ends {
                in_private_key = false;
            }
            found
        })
        .collect()
}

struct Chars<'a> {
    line: &'a str,
    offsets: Vec<usize>,
    chars: Vec<char>,
}

impl<'a> Chars<'a> {
    fn new(line: &'a str) -> Self {
        let (offsets, chars) = line.char_indices().unzip();
        Self {
            line,
            offsets,
            chars,
        }
    }

    fn len(&self) -> usize {
        self.chars.len()
    }

    fn get(&self, index: usize) -> Option<char> {
        self.chars.get(index).copied()
    }

    fn before(&self, index: usize) -> Option<char> {
        index.checked_sub(1).and_then(|index| self.get(index))
    }

    fn byte(&self, index: usize) -> usize {
        self.offsets.get(index).copied().unwrap_or(self.line.len())
    }

    fn bytes(&self, range: Range<usize>) -> Range<usize> {
        self.byte(range.start)..self.byte(range.end)
    }

    fn text(&self, range: Range<usize>) -> &'a str {
        &self.line[self.bytes(range)]
    }

    fn starts_with_at(&self, index: usize, prefix: &str) -> bool {
        let mut position = index;
        for expected in prefix.chars() {
            if self.get(position) != Some(expected) {
                return false;
            }
            position += 1;
        }
        true
    }

    fn scan_while(&self, start: usize, accept: impl Fn(char) -> bool) -> usize {
        let mut end = start;
        while self.get(end).is_some_and(&accept) {
            end += 1;
        }
        end
    }

    fn push(
        &self,
        found: &mut Vec<TextMatch>,
        kind: SensitiveKind,
        range: Range<usize>,
        key: String,
    ) {
        found.push(TextMatch {
            kind,
            range: self.bytes(range),
            key,
        });
    }
}

fn is_word(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn kind_priority(kind: SensitiveKind) -> u8 {
    match kind {
        SensitiveKind::ApiKey => 6,
        SensitiveKind::Token => 5,
        SensitiveKind::Secret => 4,
        SensitiveKind::CardNumber => 3,
        SensitiveKind::Email => 2,
        SensitiveKind::Phone => 1,
        SensitiveKind::Face => 0,
    }
}

fn merge_overlaps(line: &str, mut found: Vec<TextMatch>) -> Vec<TextMatch> {
    found.sort_by_key(|item| (item.range.start, std::cmp::Reverse(item.range.end)));
    let mut merged: Vec<TextMatch> = Vec::with_capacity(found.len());
    for item in found {
        match merged.last_mut() {
            Some(last) if item.range.start < last.range.end => {
                let union = last.range.start..last.range.end.max(item.range.end);
                if kind_priority(item.kind) > kind_priority(last.kind) {
                    last.kind = item.kind;
                }
                if union != last.range {
                    last.range = union;
                    last.key = normalize_key(last.kind, &line[last.range.clone()]);
                }
            }
            _ => merged.push(item),
        }
    }
    merged
}

fn normalize_key(kind: SensitiveKind, text: &str) -> String {
    match kind {
        SensitiveKind::Email => text.to_ascii_lowercase(),
        SensitiveKind::CardNumber | SensitiveKind::Phone => {
            text.chars().filter(char::is_ascii_digit).collect()
        }
        _ => text.to_string(),
    }
}

/// File extensions that are not also top-level domains (`.rs`, `.py`,
/// `.md`, `.zip` and `.mov` are real ones, so `milan@firma.rs` stays an
/// address).
const FILE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "svg", "webp", "avif", "heic", "ico", "bmp", "tiff", "js", "mjs",
    "cjs", "ts", "tsx", "jsx", "css", "scss", "html", "json", "txt", "rb", "java", "kt", "swift",
    "mp4", "webm", "mp3", "wav", "pdf", "lock", "toml", "yaml", "yml", "xml", "wasm", "map", "gz",
];

fn is_email_local(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '%' | '+' | '-')
}

fn is_email_domain(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '-')
}

fn find_emails(chars: &Chars, found: &mut Vec<TextMatch>) {
    for at in 0..chars.len() {
        if chars.get(at) != Some('@') {
            continue;
        }
        let mut start = at;
        while start > 0 && chars.get(start - 1).is_some_and(is_email_local) {
            start -= 1;
        }
        while start < at && matches!(chars.get(start), Some('.' | '-' | '+')) {
            start += 1;
        }
        let mut end = chars.scan_while(at + 1, is_email_domain);
        while end > at + 1 && matches!(chars.get(end - 1), Some('.' | '-')) {
            end -= 1;
        }
        let local = chars.text(start..at);
        let domain = chars.text(at + 1..end);
        let mut word_start = at;
        while word_start > 0 && !chars.get(word_start - 1).is_some_and(char::is_whitespace) {
            word_start -= 1;
        }
        let in_url = chars.text(word_start..at).contains("://");
        if !in_url && valid_email(local, domain) {
            let text = chars.text(start..end);
            chars.push(
                found,
                SensitiveKind::Email,
                start..end,
                text.to_ascii_lowercase(),
            );
        }
    }
}

fn valid_email(local: &str, domain: &str) -> bool {
    if local.is_empty() || local.len() > 64 || local.ends_with('.') || local.contains("..") {
        return false;
    }
    if local.eq_ignore_ascii_case("git") {
        return false;
    }
    let labels = domain.split('.').collect::<Vec<_>>();
    if labels.len() < 2 || domain.len() > 253 {
        return false;
    }
    if labels.iter().any(|label| {
        label.is_empty() || label.len() > 63 || label.starts_with('-') || label.ends_with('-')
    }) {
        return false;
    }
    let tld = labels[labels.len() - 1];
    let density_asset = labels.len() == 2 && is_density_suffix(labels[0]);
    (2..=24).contains(&tld.len())
        && tld.chars().all(|c| c.is_ascii_alphabetic())
        && !density_asset
        && !FILE_EXTENSIONS
            .iter()
            .any(|extension| tld.eq_ignore_ascii_case(extension))
}

/// `2x`, `3x`: the pixel-density suffix of `icon@2x.<ext>` asset names.
fn is_density_suffix(label: &str) -> bool {
    label.len() >= 2
        && label.ends_with(['x', 'X'])
        && label[..label.len() - 1].chars().all(|c| c.is_ascii_digit())
}

fn find_cards(chars: &Chars, found: &mut Vec<TextMatch>) {
    let mut index = 0;
    while index < chars.len() {
        let starts_run = chars.get(index).is_some_and(|c| c.is_ascii_digit())
            && !chars
                .before(index)
                .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | ',' | '+'));
        if !starts_run {
            index += 1;
            continue;
        }

        let mut groups: Vec<Range<usize>> = Vec::new();
        let mut separator = None;
        let mut cursor = index;
        loop {
            let end = chars.scan_while(cursor, |c| c.is_ascii_digit());
            groups.push(cursor..end);
            let Some(next) = chars.get(end) else {
                break;
            };
            let continues = matches!(next, ' ' | '-')
                && separator.is_none_or(|separator| separator == next)
                && chars.get(end + 1).is_some_and(|c| c.is_ascii_digit());
            if !continues {
                break;
            }
            separator = Some(next);
            cursor = end + 1;
        }

        let run_end = groups.last().map_or(index, |group| group.end);
        let trailing_ok = chars
            .get(run_end)
            .is_none_or(|c| !c.is_ascii_alphabetic() && c != '_')
            && !(matches!(chars.get(run_end), Some('.' | ','))
                && chars.get(run_end + 1).is_some_and(|c| c.is_ascii_digit()));

        if let Some(card) = best_card_in_groups(chars, &groups, trailing_ok) {
            let digits = chars
                .text(card.clone())
                .chars()
                .filter(char::is_ascii_digit)
                .collect::<String>();
            chars.push(found, SensitiveKind::CardNumber, card, digits);
        }
        index = run_end.max(index + 1);
    }
}

fn best_card_in_groups(
    chars: &Chars,
    groups: &[Range<usize>],
    trailing_ok: bool,
) -> Option<Range<usize>> {
    for first in 0..groups.len() {
        for last in (first..groups.len()).rev() {
            if last == groups.len() - 1 && !trailing_ok {
                continue;
            }
            let selected = &groups[first..=last];
            let sizes = selected.iter().map(|group| group.len()).collect::<Vec<_>>();
            if !card_grouping(&sizes) {
                continue;
            }
            let digits = selected
                .iter()
                .flat_map(|group| chars.text(group.clone()).chars())
                .collect::<String>();
            if is_card_number(&digits) {
                return Some(selected[0].start..selected[selected.len() - 1].end);
            }
        }
    }
    None
}

fn card_grouping(sizes: &[usize]) -> bool {
    let total = sizes.iter().sum::<usize>();
    if !(13..=19).contains(&total) {
        return false;
    }
    match sizes {
        [_] => true,
        [4, 6, 5] | [4, 6, 4] => true,
        [init @ .., last] => init.iter().all(|size| *size == 4) && (1..=4).contains(last),
        [] => false,
    }
}

pub fn luhn_valid(digits: &str) -> bool {
    let mut sum = 0;
    for (position, c) in digits.chars().rev().enumerate() {
        let Some(mut digit) = c.to_digit(10) else {
            return false;
        };
        if position % 2 == 1 {
            digit *= 2;
            if digit > 9 {
                digit -= 9;
            }
        }
        sum += digit;
    }
    !digits.is_empty() && sum % 10 == 0
}

fn is_card_number(digits: &str) -> bool {
    let len = digits.len();
    if !(13..=19).contains(&len) || !luhn_valid(digits) {
        return false;
    }
    if digits.bytes().all(|b| b == digits.as_bytes()[0]) {
        return false;
    }
    let prefix = |n: usize| digits[..n].parse::<u32>().unwrap_or(0);
    match prefix(1) {
        4 => matches!(len, 13 | 16 | 19),
        5 => (51..=55).contains(&prefix(2)) && len == 16 || matches!(prefix(2), 50 | 56..=59),
        2 => (2221..=2720).contains(&prefix(4)) && len == 16,
        3 => {
            matches!(prefix(2), 34 | 37) && len == 15
                || (3528..=3589).contains(&prefix(4)) && len >= 16
                || ((300..=305).contains(&prefix(3)) || matches!(prefix(2), 36 | 38 | 39))
                    && len >= 14
        }
        6 => len >= 16,
        _ => false,
    }
}

struct KeyPattern {
    prefix: &'static str,
    min_tail: usize,
    max_tail: usize,
    charset: fn(char) -> bool,
    needs_digit: bool,
}

impl KeyPattern {
    const fn new(
        prefix: &'static str,
        min_tail: usize,
        max_tail: usize,
        charset: fn(char) -> bool,
        needs_digit: bool,
    ) -> Self {
        Self {
            prefix,
            min_tail,
            max_tail,
            charset,
            needs_digit,
        }
    }
}

fn key_body(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

fn alnum(c: char) -> bool {
    c.is_ascii_alphanumeric()
}

fn upper_alnum(c: char) -> bool {
    c.is_ascii_uppercase() || c.is_ascii_digit()
}

fn hex(c: char) -> bool {
    c.is_ascii_hexdigit()
}

fn sendgrid(c: char) -> bool {
    key_body(c) || c == '.'
}

const KEY_PATTERNS: &[KeyPattern] = &[
    KeyPattern::new("sk-", 20, 256, key_body, true),
    KeyPattern::new("sk_live_", 16, 256, alnum, false),
    KeyPattern::new("sk_test_", 16, 256, alnum, false),
    KeyPattern::new("rk_live_", 16, 256, alnum, false),
    KeyPattern::new("rk_test_", 16, 256, alnum, false),
    KeyPattern::new("whsec_", 24, 256, alnum, false),
    KeyPattern::new("ghp_", 30, 255, alnum, false),
    KeyPattern::new("gho_", 30, 255, alnum, false),
    KeyPattern::new("ghu_", 30, 255, alnum, false),
    KeyPattern::new("ghs_", 30, 255, alnum, false),
    KeyPattern::new("ghr_", 30, 255, alnum, false),
    KeyPattern::new("github_pat_", 30, 255, key_body, false),
    KeyPattern::new("glpat-", 20, 255, key_body, false),
    KeyPattern::new("xoxb-", 10, 255, key_body, true),
    KeyPattern::new("xoxp-", 10, 255, key_body, true),
    KeyPattern::new("xoxa-", 10, 255, key_body, true),
    KeyPattern::new("xoxr-", 10, 255, key_body, true),
    KeyPattern::new("xoxs-", 10, 255, key_body, true),
    KeyPattern::new("xoxe-", 10, 255, key_body, true),
    KeyPattern::new("xapp-", 10, 255, key_body, true),
    KeyPattern::new("AKIA", 16, 16, upper_alnum, false),
    KeyPattern::new("ASIA", 16, 16, upper_alnum, false),
    KeyPattern::new("AIza", 35, 35, key_body, false),
    KeyPattern::new("npm_", 36, 36, alnum, false),
    KeyPattern::new("hf_", 30, 64, alnum, false),
    KeyPattern::new("SG.", 40, 80, sendgrid, false),
    KeyPattern::new("shpat_", 32, 32, hex, false),
    KeyPattern::new("shpss_", 32, 32, hex, false),
    KeyPattern::new("dop_v1_", 64, 64, hex, false),
    KeyPattern::new("pypi-", 50, 255, key_body, false),
];

fn find_prefixed_keys(chars: &Chars, found: &mut Vec<TextMatch>) {
    for start in 0..chars.len() {
        if chars.before(start).is_some_and(is_word) {
            continue;
        }
        for pattern in KEY_PATTERNS {
            if !chars.starts_with_at(start, pattern.prefix) {
                continue;
            }
            let tail_start = start + pattern.prefix.chars().count();
            let end = chars.scan_while(tail_start, pattern.charset);
            let tail = end - tail_start;
            if tail < pattern.min_tail || tail > pattern.max_tail {
                continue;
            }
            let text = chars.text(start..end);
            if pattern.needs_digit
                && !chars
                    .text(tail_start..end)
                    .contains(|c: char| c.is_ascii_digit())
            {
                continue;
            }
            chars.push(found, SensitiveKind::ApiKey, start..end, text.to_string());
            break;
        }
    }
}

fn base64url(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

fn find_jwts(chars: &Chars, found: &mut Vec<TextMatch>) {
    let mut start = 0;
    while start < chars.len() {
        if !chars.starts_with_at(start, "eyJ") || chars.before(start).is_some_and(base64url) {
            start += 1;
            continue;
        }
        let header_end = chars.scan_while(start, base64url);
        if header_end - start < 10 || chars.get(header_end) != Some('.') {
            start = header_end.max(start + 1);
            continue;
        }
        let payload_end = chars.scan_while(header_end + 1, base64url);
        if payload_end - (header_end + 1) < 10 {
            start = payload_end.max(start + 1);
            continue;
        }
        let mut end = payload_end;
        if chars.get(payload_end) == Some('.') {
            let signature_end = chars.scan_while(payload_end + 1, base64url);
            if signature_end > payload_end + 1 {
                end = signature_end;
            }
        }
        let text = chars.text(start..end);
        chars.push(found, SensitiveKind::Token, start..end, text.to_string());
        start = end;
    }
}

/// Telegram bot tokens, `<bot id>:<35 characters>`, also inside
/// `api.telegram.org/bot<token>/...` URLs.
fn find_telegram_tokens(chars: &Chars, found: &mut Vec<TextMatch>) {
    let mut start = 0;
    while start < chars.len() {
        let starts_digits = chars.get(start).is_some_and(|c| c.is_ascii_digit())
            && !chars.before(start).is_some_and(|c| c.is_ascii_digit())
            && (!chars
                .before(start)
                .is_some_and(|c| c.is_ascii_alphanumeric())
                || start >= 3 && chars.text(start - 3..start).eq_ignore_ascii_case("bot"));
        if !starts_digits {
            start += 1;
            continue;
        }
        let digits_end = chars.scan_while(start, |c| c.is_ascii_digit());
        let id_len = digits_end - start;
        if (6..=12).contains(&id_len) && chars.get(digits_end) == Some(':') {
            let end = chars.scan_while(digits_end + 1, key_body);
            if end - (digits_end + 1) >= 30 {
                let text = chars.text(start..end);
                chars.push(found, SensitiveKind::ApiKey, start..end, text.to_string());
                start = end;
                continue;
            }
        }
        start = digits_end.max(start + 1);
    }
}

fn token_body(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_' | '~' | '+' | '/' | '=')
}

fn find_bearer_tokens(chars: &Chars, found: &mut Vec<TextMatch>) {
    let lower = chars.line.to_ascii_lowercase();
    let mut search = 0;
    while let Some(position) = lower[search..].find("bearer") {
        let byte = search + position;
        search = byte + "bearer".len();
        let Some(index) = chars.offsets.iter().position(|offset| *offset == byte) else {
            continue;
        };
        if chars.before(index).is_some_and(is_word) {
            continue;
        }
        let after = index + "bearer".len();
        let value_start = chars.scan_while(after, |c| c == ' ');
        if value_start == after {
            continue;
        }
        let end = chars.scan_while(value_start, token_body);
        let value = chars.text(value_start..end);
        if value.len() >= 20 && has_letter_and_digit(value) {
            chars.push(
                found,
                SensitiveKind::Token,
                value_start..end,
                value.to_string(),
            );
        }
    }
}

const SECRET_LABELS: &[&str] = &[
    "password",
    "passwd",
    "passphrase",
    "secret",
    "token",
    "apikey",
    "api_key",
    "api-key",
    "access_key",
    "access-key",
    "accesskey",
    "private_key",
    "private-key",
    "client_secret",
    "auth",
    "credential",
];

/// Labels whose value is secret whatever it looks like: a password of plain
/// words is still a password.
const ANY_VALUE_LABELS: &[&str] = &[
    "password",
    "passwd",
    "passphrase",
    "secret",
    "token",
    "client_secret",
];

const PLACEHOLDERS: &[&str] = &[
    "null",
    "none",
    "nil",
    "true",
    "false",
    "undefined",
    "required",
    "optional",
    "redacted",
    "hidden",
    "secret",
    "password",
    "changeme",
];

/// What follows `password:` in code when it is a type, not a value.
const TYPE_NAMES: &[&str] = &[
    "string",
    "str",
    "&str",
    "&'static",
    "number",
    "bool",
    "boolean",
    "int",
    "integer",
    "i32",
    "i64",
    "u8",
    "u32",
    "u64",
    "usize",
    "char",
    "byte",
    "bytes",
    "any",
    "unknown",
    "object",
    "securestring",
    "secret",
    "option",
    "vec",
    "text",
    "varchar",
];

fn is_value_char(c: char) -> bool {
    !c.is_whitespace()
        && !matches!(
            c,
            '"' | '\'' | '`' | ',' | ';' | ')' | ']' | '}' | '>' | '&'
        )
}

/// `password: hunter2`, `API_KEY=...`, `"token": "..."`: a value written
/// right after a credential-looking label.
fn find_labeled_secrets(chars: &Chars, found: &mut Vec<TextMatch>) {
    let lower = chars.line.to_ascii_lowercase();
    for label in SECRET_LABELS {
        let mut search = 0;
        while let Some(position) = lower[search..].find(label) {
            let byte = search + position;
            search = byte + label.len();
            let Some(index) = chars.offsets.iter().position(|offset| *offset == byte) else {
                continue;
            };
            if chars
                .before(index)
                .is_some_and(|c| c.is_ascii_alphanumeric())
            {
                continue;
            }
            let mut cursor = index + label.chars().count();
            if matches!(chars.get(cursor), Some('_' | '-')) {
                cursor = chars.scan_while(cursor, |c| is_word(c) || c == '-');
            }
            cursor = chars.scan_while(cursor, |c| matches!(c, ' ' | '"' | '\''));
            if !matches!(chars.get(cursor), Some(':' | '=')) {
                continue;
            }
            cursor = chars.scan_while(cursor + 1, |c| matches!(c, ' ' | '"' | '\'' | '`'));
            let end = chars.scan_while(cursor, is_value_char);
            let value = chars.text(cursor..end);
            let any_value = ANY_VALUE_LABELS.contains(label);
            if looks_like_secret_value(value, chars.get(end), any_value) {
                chars.push(found, SensitiveKind::Secret, cursor..end, value.to_string());
            }
        }
    }
}

fn is_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_pascal_case(text: &str) -> bool {
    text.starts_with(|c: char| c.is_ascii_uppercase())
        && text.chars().all(|c| c.is_ascii_alphabetic())
        && text.chars().any(|c| c.is_ascii_lowercase())
}

/// Whether the text after a credential label is a value rather than code:
/// a type (`password: SecureString,`, `Option<String>`), a call
/// (`getPassword()`), an environment reference (`${TOKEN}`, `$API_KEY`,
/// `process.env.X`) or a placeholder.
fn looks_like_secret_value(value: &str, next: Option<char>, any_value: bool) -> bool {
    if value.is_empty() || value.len() > 512 {
        return false;
    }
    let lower = value.to_ascii_lowercase();
    if PLACEHOLDERS.contains(&lower.as_str())
        || TYPE_NAMES.contains(&lower.as_str())
        || lower.starts_with("your")
        || lower
            .chars()
            .all(|c| matches!(c, 'x' | '*' | '•' | '.' | '-' | '_'))
    {
        return false;
    }
    if value.starts_with(['{', '[', '<'])
        || value.starts_with("${")
        || value.starts_with("$(")
        || value.contains("process.env")
        || value.contains("environ")
        || value.contains("://")
        || value.contains("::")
    {
        return false;
    }
    let env_name = |name: &str| {
        !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
    };
    if value.strip_prefix('$').is_some_and(env_name)
        || value.len() > 2
            && value.starts_with('%')
            && value.ends_with('%')
            && env_name(&value[1..value.len() - 1])
    {
        return false;
    }
    if env_name(value) && value.contains('_') {
        return false;
    }
    if let Some(open) = value.find(['(', '<']) {
        let head = &value[..open];
        let call = value.ends_with('(') && head.split('.').all(is_identifier);
        let generic = value[open..].starts_with('<') && is_pascal_case(head);
        if call || generic {
            return false;
        }
    }
    if is_pascal_case(value) && matches!(next, Some(',' | ';' | ')' | '>')) {
        return false;
    }
    let dotted_path = value.split('.').count() > 1
        && value.split('.').all(|part| {
            !part.is_empty() && part.chars().all(|c| c.is_ascii_alphabetic() || c == '_')
        });
    if dotted_path {
        return false;
    }
    if any_value {
        return !(value.chars().all(|c| c.is_ascii_digit()) && value.len() < 6);
    }
    let classes = [
        value.chars().any(|c| c.is_ascii_lowercase()),
        value.chars().any(|c| c.is_ascii_uppercase()),
        value.chars().any(|c| c.is_ascii_digit()),
        value.chars().any(|c| !c.is_ascii_alphanumeric()),
    ]
    .into_iter()
    .filter(|present| *present)
    .count();
    value.chars().count() >= 8 && classes >= 2 && has_letter_and_digit(value)
}

fn has_letter_and_digit(value: &str) -> bool {
    value.chars().any(|c| c.is_ascii_alphabetic()) && value.chars().any(|c| c.is_ascii_digit())
}

const CONTEXT_WORDS: &[&str] = &[
    "secret",
    "secrets",
    "token",
    "tokens",
    "key",
    "keys",
    "apikey",
    "accesskey",
    "privatekey",
    "secretkey",
    "password",
    "passwd",
    "auth",
    "authorization",
    "bearer",
    "credential",
    "credentials",
    "signature",
    "sig",
];

const URL_SECRET_PARAMS: &[&str] = &[
    "token",
    "access_token",
    "refresh_token",
    "id_token",
    "key",
    "api_key",
    "apikey",
    "secret",
    "client_secret",
    "signature",
    "sig",
    "x-amz-signature",
    "x-amz-credential",
    "x-amz-security-token",
    "x-goog-signature",
    "x-goog-credential",
    "googleaccessid",
    "password",
    "auth",
    "code",
];

/// SRI hashes (`integrity: sha512-...`) are long base64 but not secret.
const DIGEST_PREFIXES: &[&str] = &["sha1-", "sha256-", "sha384-", "sha512-"];

fn is_b64(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '_' | '-')
}

/// Whether `text` names a credential as a whole word (`api_key`, `apiKey`,
/// `token:`), never inside another word (`monkey`, `keyboard`).
fn has_context_word(text: &str) -> bool {
    let mut words = Vec::new();
    for part in text.split(|c: char| !c.is_ascii_alphanumeric()) {
        let mut start = 0;
        let bytes = part.as_bytes();
        for index in 1..bytes.len() {
            if bytes[index - 1].is_ascii_lowercase() && bytes[index].is_ascii_uppercase() {
                words.push(&part[start..index]);
                start = index;
            }
        }
        words.push(&part[start..]);
    }
    words.iter().any(|word| {
        CONTEXT_WORDS
            .iter()
            .any(|context| word.eq_ignore_ascii_case(context))
    })
}

/// Long random-looking strings: base64 blobs and, only when the line says
/// it is a credential, hex. Plain hashes (commit SHAs, digests, asset
/// fingerprints) are everywhere on a developer's screen and are not secret,
/// so a bare hex run never counts and a URL only counts through credentials
/// it carries.
fn find_high_entropy(chars: &Chars, found: &mut Vec<TextMatch>) {
    let mut index = 0;
    while index < chars.len() {
        if chars.get(index).is_some_and(char::is_whitespace) {
            index += 1;
            continue;
        }
        let word_end = chars.scan_while(index, |c| !c.is_whitespace());
        let word = index..word_end;
        let text = chars.text(word.clone());
        let lower = text.to_ascii_lowercase();
        if lower.contains("data:") && lower.contains(";base64,") {
            index = word_end;
            continue;
        }
        if text.contains("://") || text.starts_with("www.") {
            find_url_secrets(chars, word.clone(), found);
        } else {
            let mut cursor = word.start;
            while cursor < word.end {
                if !chars.get(cursor).is_some_and(is_b64) {
                    cursor += 1;
                    continue;
                }
                let body_end = chars.scan_while(cursor, is_b64).min(word.end);
                let padding_end = chars.scan_while(body_end, |c| c == '=').min(word.end);
                let end = if padding_end - body_end <= 2 {
                    padding_end
                } else {
                    body_end
                };
                let mut start = cursor;
                while start < end && matches!(chars.get(start), Some('/' | '-' | '_')) {
                    start += 1;
                }
                let candidate = chars.text(start..end);
                let in_context = has_context_word(&chars.line[..chars.byte(start)]);
                if random_looking(candidate, in_context) {
                    chars.push(
                        found,
                        SensitiveKind::Secret,
                        start..end,
                        candidate.to_string(),
                    );
                }
                cursor = end.max(cursor + 1);
            }
        }
        index = word_end;
    }
}

/// A whole line of base64 the width PEM and similar armoured key formats
/// wrap at: the body of a private key, even when its header is off screen.
fn find_pem_body(chars: &Chars, found: &mut Vec<TextMatch>) {
    let trimmed = chars.line.trim();
    let body = trimmed.trim_end_matches('=');
    let armoured = (60..=80).contains(&trimmed.len())
        && trimmed.len() - body.len() <= 2
        && body
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/')
        && body.chars().any(|c| c.is_ascii_digit())
        && body.chars().any(|c| c.is_ascii_uppercase())
        && body.chars().any(|c| c.is_ascii_lowercase())
        && shannon_entropy(body) >= 4.0;
    if armoured {
        let start = chars.line.len() - chars.line.trim_start().len();
        found.push(TextMatch {
            kind: SensitiveKind::Secret,
            range: start..start + trimmed.len(),
            key: trimmed.to_string(),
        });
    }
}

fn find_url_secrets(chars: &Chars, word: Range<usize>, found: &mut Vec<TextMatch>) {
    let text = chars.text(word.clone());
    let char_at = |byte: usize| word.start + text[..byte].chars().count();
    let mut push = |start: usize, end: usize| {
        if end > start {
            let range = char_at(start)..char_at(end);
            let value = chars.text(range.clone());
            chars.push(found, SensitiveKind::Secret, range, value.to_string());
        }
    };

    if let Some(scheme_end) = text.find("://") {
        let authority_start = scheme_end + 3;
        let authority_end = text[authority_start..]
            .find(['/', '?', '#'])
            .map_or(text.len(), |offset| authority_start + offset);
        let authority = &text[authority_start..authority_end];
        if let Some(at) = authority.rfind('@')
            && let Some(colon) = authority[..at].find(':')
        {
            push(authority_start + colon + 1, authority_start + at);
        }
        let path_end = text[authority_end..]
            .find(['?', '#'])
            .map_or(text.len(), |offset| authority_end + offset);
        let path = &text[authority_end..path_end];
        let host = authority
            .rsplit('@')
            .next()
            .unwrap_or(authority)
            .to_ascii_lowercase();
        for (host_part, marker) in [
            ("hooks.slack.com", "/services/"),
            ("discord.com", "/api/webhooks/"),
            ("discordapp.com", "/api/webhooks/"),
        ] {
            if host.contains(host_part)
                && let Some(offset) = path.find(marker)
            {
                push(authority_end + offset + marker.len(), path_end);
            }
        }
    }

    let mut cursor = word.start;
    while cursor < word.end {
        if !matches!(chars.get(cursor), Some('?' | '&' | '#' | ';')) {
            cursor += 1;
            continue;
        }
        let name_start = cursor + 1;
        let name_end = chars.scan_while(name_start, |c| c != '=' && c != '&' && c != '#');
        if chars.get(name_end) != Some('=') || name_end > word.end {
            cursor = name_start;
            continue;
        }
        let value_start = name_end + 1;
        let value_end = chars
            .scan_while(value_start, |c| {
                !matches!(c, '&' | '#' | ';') && !c.is_whitespace()
            })
            .min(word.end);
        let name = chars.text(name_start..name_end).to_ascii_lowercase();
        let value = chars.text(value_start..value_end);
        if URL_SECRET_PARAMS.contains(&name.as_str())
            && value.len() >= 12
            && has_letter_and_digit(value)
        {
            chars.push(
                found,
                SensitiveKind::Secret,
                value_start..value_end,
                value.to_string(),
            );
        }
        cursor = value_end.max(cursor + 1);
    }
}

pub fn shannon_entropy(text: &str) -> f64 {
    let mut counts = [0u32; 128];
    let mut total = 0u32;
    for byte in text.bytes() {
        counts[(byte & 0x7f) as usize] += 1;
        total += 1;
    }
    if total == 0 {
        return 0.0;
    }
    counts
        .iter()
        .filter(|count| **count > 0)
        .map(|count| {
            let p = f64::from(*count) / f64::from(total);
            -p * p.log2()
        })
        .sum()
}

fn longest_run(text: &str, class: impl Fn(char) -> bool) -> usize {
    let mut longest = 0;
    let mut current = 0;
    for c in text.chars() {
        if class(c) {
            current += 1;
            longest = longest.max(current);
        } else {
            current = 0;
        }
    }
    longest
}

/// A path is made of words: each segment reads as lowercase, Capitalized or
/// UPPERCASE. Random base64 with slashes mixes case inside its segments.
fn path_like(candidate: &str) -> bool {
    candidate.matches('/').count() >= 2
        && candidate
            .split('/')
            .filter(|part| !part.is_empty())
            .all(|part| {
                let rest = part.get(1..).unwrap_or("");
                !rest.chars().any(|c| c.is_ascii_uppercase())
                    || !part.chars().any(|c| c.is_ascii_lowercase())
            })
}

fn random_looking(candidate: &str, in_context: bool) -> bool {
    let len = candidate.len();
    let lower = candidate.to_ascii_lowercase();
    if DIGEST_PREFIXES
        .iter()
        .any(|prefix| lower.starts_with(prefix))
    {
        return false;
    }
    let is_hex = candidate.chars().all(|c| c.is_ascii_hexdigit() || c == '-');
    if is_hex {
        let digits = candidate.chars().filter(char::is_ascii_hexdigit).count();
        return in_context
            && digits >= 32
            && candidate.chars().any(|c| c.is_ascii_digit())
            && candidate.chars().any(|c| c.is_ascii_alphabetic());
    }
    if candidate.chars().all(|c| c.is_ascii_digit()) || path_like(candidate) {
        return false;
    }
    let has_lower = candidate.chars().any(|c| c.is_ascii_lowercase());
    let has_upper = candidate.chars().any(|c| c.is_ascii_uppercase());
    let digits = candidate.chars().filter(char::is_ascii_digit).count();
    let entropy = shannon_entropy(candidate);
    if in_context {
        return len >= 20 && has_letter_and_digit(candidate) && entropy >= 3.5;
    }
    let long = len >= 40;
    // Long camelCase identifiers are the false positive to beat: their
    // lowercase runs are whole words, while in random base64 they average
    // under two letters and almost never reach ten.
    let jagged = longest_run(candidate, |c| c.is_ascii_lowercase()) <= 9
        && longest_run(candidate, |c| c.is_ascii_uppercase()) <= 9;
    let wordy = mean_run(candidate, |c| c.is_ascii_lowercase()) >= 3.0;
    len >= 32
        && has_lower
        && has_upper
        && digits >= if long { 1 } else { 2 }
        && entropy >= 4.0
        && (long || jagged)
        && !wordy
}

fn mean_run(text: &str, class: impl Fn(char) -> bool) -> f64 {
    let mut runs = 0usize;
    let mut members = 0usize;
    let mut previous = false;
    for c in text.chars() {
        let member = class(c);
        if member {
            members += 1;
            if !previous {
                runs += 1;
            }
        }
        previous = member;
    }
    if runs == 0 {
        0.0
    } else {
        members as f64 / runs as f64
    }
}

fn find_phones(chars: &Chars, found: &mut Vec<TextMatch>) {
    let mut index = 0;
    while index < chars.len() {
        let boundary = !chars
            .before(index)
            .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '/' | '+'));
        if !boundary {
            index += 1;
            continue;
        }
        let matched = if chars.get(index) == Some('+') {
            international_phone(chars, index)
        } else {
            nanp_phone(chars, index)
        };
        match matched {
            Some(end) => {
                let digits = chars
                    .text(index..end)
                    .chars()
                    .filter(char::is_ascii_digit)
                    .collect();
                chars.push(found, SensitiveKind::Phone, index..end, digits);
                index = end;
            }
            None => index += 1,
        }
    }
}

fn phone_tail_ok(chars: &Chars, end: usize) -> bool {
    chars
        .get(end)
        .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_' && c != '@')
        && !(matches!(chars.get(end), Some('.' | '-' | '/' | ':'))
            && chars.get(end + 1).is_some_and(|c| c.is_ascii_digit()))
}

fn international_phone(chars: &Chars, start: usize) -> Option<usize> {
    if !chars
        .get(start + 1)
        .is_some_and(|c| c.is_ascii_digit() && c != '0')
    {
        return None;
    }
    let mut digits = 0;
    let mut cursor = start + 1;
    let mut last_digit_end = cursor;
    let mut previous_separator = false;
    let mut open_paren = false;
    while let Some(c) = chars.get(cursor) {
        if c.is_ascii_digit() {
            digits += 1;
            previous_separator = false;
            last_digit_end = cursor + 1;
        } else if matches!(c, ' ' | '-' | '.') && !previous_separator {
            previous_separator = true;
        } else if c == '(' && !open_paren {
            open_paren = true;
        } else if c == ')' && open_paren {
            open_paren = false;
            previous_separator = false;
        } else {
            break;
        }
        if digits > 15 {
            return None;
        }
        cursor += 1;
    }
    let separators = chars
        .text(start + 1..last_digit_end)
        .chars()
        .filter(|c| !c.is_ascii_digit())
        .count();
    ((8..=15).contains(&digits)
        && (separators > 0 || digits >= 10)
        && phone_tail_ok(chars, last_digit_end))
    .then_some(last_digit_end)
}

fn digits_at(chars: &Chars, start: usize, count: usize) -> bool {
    (start..start + count).all(|index| chars.get(index).is_some_and(|c| c.is_ascii_digit()))
}

fn nanp_phone(chars: &Chars, start: usize) -> Option<usize> {
    let (exchange_start, separator) = if chars.get(start) == Some('(') {
        if !digits_at(chars, start + 1, 3) || chars.get(start + 4) != Some(')') {
            return None;
        }
        let mut cursor = start + 5;
        if chars.get(cursor) == Some(' ') {
            cursor += 1;
        }
        (cursor, None)
    } else {
        if !digits_at(chars, start, 3) {
            return None;
        }
        let separator = chars.get(start + 3)?;
        if !matches!(separator, '-' | '.' | ' ') {
            return None;
        }
        (start + 4, Some(separator))
    };
    let area = chars.text(start..exchange_start);
    let area_starts_high = area
        .chars()
        .find(|c| c.is_ascii_digit())
        .is_some_and(|c| c >= '2');
    let exchange_starts_high = chars.get(exchange_start).is_some_and(|c| c >= '2');
    if !area_starts_high || !digits_at(chars, exchange_start, 3) || !exchange_starts_high {
        return None;
    }
    let second = chars.get(exchange_start + 3)?;
    let second_ok = match separator {
        Some(separator) => second == separator,
        None => matches!(second, '-' | '.' | ' '),
    };
    if !second_ok || !digits_at(chars, exchange_start + 4, 4) {
        return None;
    }
    let end = exchange_start + 8;
    phone_tail_ok(chars, end).then_some(end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(line: &str) -> Vec<(SensitiveKind, &str)> {
        find_sensitive_text(line, &DetectorOptions::default())
            .into_iter()
            .map(|item| (item.kind, &line[item.range]))
            .collect()
    }

    fn kinds(line: &str) -> Vec<SensitiveKind> {
        found(line).into_iter().map(|(kind, _)| kind).collect()
    }

    #[test]
    fn finds_emails_with_their_exact_span() {
        assert_eq!(
            found("Contact: jane.doe@example.com (support)"),
            [(SensitiveKind::Email, "jane.doe@example.com")]
        );
        assert_eq!(
            found("<First.Last+tag@mail.co.uk>,"),
            [(SensitiveKind::Email, "First.Last+tag@mail.co.uk")]
        );
        assert_eq!(
            found("Author: AJ <aj_dev-99@sub.example.io>"),
            [(SensitiveKind::Email, "aj_dev-99@sub.example.io")]
        );
        assert_eq!(
            found("mail me at a@b.co."),
            [(SensitiveKind::Email, "a@b.co")]
        );
        let key = &find_sensitive_text("JANE@Example.COM", &DetectorOptions::default())[0].key;
        assert_eq!(key, "jane@example.com");
    }

    #[test]
    fn ignores_things_that_only_look_like_emails() {
        for line in [
            "npm i react@18.3.1",
            "icon@2x.png and logo@3x.webp",
            "git@github.com:CapSoftware/Cap.git",
            "@types/node and @scope/package",
            "user@localhost",
            "meet @ 10am",
            "foo@bar",
            "a@b.c",
            "x@-bad.com",
        ] {
            assert!(
                !kinds(line).contains(&SensitiveKind::Email),
                "{line:?}: {:?}",
                found(line)
            );
        }
    }

    #[test]
    fn luhn_matches_known_numbers() {
        assert!(luhn_valid("4242424242424242"));
        assert!(luhn_valid("79927398713"));
        assert!(!luhn_valid("4242424242424241"));
        assert!(!luhn_valid(""));
        assert!(!luhn_valid("42a2"));
    }

    #[test]
    fn finds_card_numbers_in_common_groupings() {
        for (line, card) in [
            (
                "Card: 4242 4242 4242 4242   exp 12/29",
                "4242 4242 4242 4242",
            ),
            ("4000-0566-5566-5556", "4000-0566-5566-5556"),
            ("number 5555555555554444.", "5555555555554444"),
            ("amex 3782 822463 10005 cvc 1234", "3782 822463 10005"),
            ("Mastercard 2223 0031 2200 3222", "2223 0031 2200 3222"),
            ("discover: 6011111111111117", "6011111111111117"),
            ("Qty 2 4242 4242 4242 4242", "4242 4242 4242 4242"),
        ] {
            assert_eq!(found(line), [(SensitiveKind::CardNumber, card)], "{line:?}");
        }
        let key = &find_sensitive_text("4242-4242-4242-4242", &DetectorOptions::default())[0].key;
        assert_eq!(key, "4242424242424242");
    }

    #[test]
    fn ignores_numbers_that_are_not_cards() {
        for line in [
            "4242 4242 4242 4241",
            "timestamp 1698765432123 ms",
            "order #100234  build 20240115",
            "released 2024-01-15 at 10:30",
            "id 0000000000000000",
            "v4242424242424242",
            "4242424242424242abc",
            "3.14159265358979323",
            "4242 4242-4242 4242",
            "IBAN 1234 5678 9012 3456 7890 12",
        ] {
            assert!(
                !kinds(line).contains(&SensitiveKind::CardNumber),
                "{line:?}: {:?}",
                found(line)
            );
        }
    }

    #[test]
    fn finds_vendor_keys_by_prefix() {
        for (line, key) in [
            (
                concat!("OPENAI_API_KEY=sk", "-proj-Xr7kQ2mN9vB4tL8wZ1cY6pH3"),
                concat!("sk", "-proj-Xr7kQ2mN9vB4tL8wZ1cY6pH3"),
            ),
            (
                concat!("export ANTHROPIC=sk", "-ant-api03-AbCdEf123456GhIjKl7890"),
                concat!("sk", "-ant-api03-AbCdEf123456GhIjKl7890"),
            ),
            (
                concat!("token gh", "p_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"),
                concat!("gh", "p_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"),
            ),
            (
                concat!(
                    "gi",
                    "thub_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz"
                ),
                concat!(
                    "gi",
                    "thub_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz"
                ),
            ),
            (
                concat!("aws AK", "IAIOSFODNN7EXAMPLE here"),
                concat!("AK", "IAIOSFODNN7EXAMPLE"),
            ),
            (
                concat!("SLACK=xo", "xb-1234567890-0987654321-AbCdEfGhIjKl"),
                concat!("xo", "xb-1234567890-0987654321-AbCdEfGhIjKl"),
            ),
            (
                concat!("sk", "_live_51HxYzAbCdEfGhIjKlMnOpQr"),
                concat!("sk", "_live_51HxYzAbCdEfGhIjKlMnOpQr"),
            ),
            (
                concat!("key=AI", "zaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"),
                concat!("AI", "zaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"),
            ),
        ] {
            let items = found(line);
            assert!(
                items.contains(&(SensitiveKind::ApiKey, key)),
                "{line:?}: {items:?}"
            );
        }
    }

    #[test]
    fn prefixes_need_a_boundary_and_a_real_body() {
        for line in [
            "sk-learn is a python package",
            "risk-adjusted-return-calculation-model-v2",
            concat!("task-gh", "p_notreallyatoken"),
            "AKIA is the AWS access key prefix",
            concat!("AK", "IAIOSFODNN7EXAMPLE1234"),
            concat!("xo", "xb-placeholder"),
        ] {
            assert!(
                !kinds(line).contains(&SensitiveKind::ApiKey),
                "{line:?}: {:?}",
                found(line)
            );
        }
    }

    #[test]
    fn finds_jwts_and_bearer_tokens() {
        let jwt = concat!(
            "ey",
            "JhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        );
        assert_eq!(
            found(&format!("Authorization: {jwt}")),
            [(SensitiveKind::Token, jwt)]
        );
        assert_eq!(
            found("curl -H 'Authorization: Bearer abc123DEF456ghi789JKL0'"),
            [(SensitiveKind::Token, "abc123DEF456ghi789JKL0")]
        );
        assert!(found("the bearer of bad news").is_empty());
        assert!(found("eyJ is how base64 JSON starts").is_empty());
    }

    #[test]
    fn finds_values_written_after_credential_labels() {
        for (line, value) in [
            ("password: hunter2!", "hunter2!"),
            ("DB_PASSWORD=Sup3rS3cret", "Sup3rS3cret"),
            ("\"client_secret\": \"a8F3kL9qP2xZ\"", "a8F3kL9qP2xZ"),
            ("GITHUB_TOKEN = 7f3k29dk1lq0x", "7f3k29dk1lq0x"),
        ] {
            let items = found(line);
            assert!(
                items.iter().any(|(_, text)| *text == value),
                "{line:?}: {items:?}"
            );
        }
    }

    #[test]
    fn ignores_labels_without_a_secret_value() {
        for line in [
            "token: string",
            "password: Option<String>,",
            "apiKey: process.env.API_KEY",
            "API_KEY=${API_KEY}",
            "secret = os.environ['SECRET']",
            "password: ********",
            "token: <your token here>",
            "Author: John Smith",
            "tokens: 1024",
            "passwordless sign in",
            "auth: config.auth.provider",
        ] {
            assert!(found(line).is_empty(), "{line:?}: {:?}", found(line));
        }
    }

    #[test]
    fn finds_long_random_strings() {
        let blob = "Zm9vYmFyQmF6UXV4MTIzNDU2Nzg5MGFiY2RlZkdISUpLTE1O";
        assert_eq!(found(blob), [(SensitiveKind::Secret, blob)]);
        let hex = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";
        assert_eq!(
            found(&format!("SECRET_KEY {hex}")),
            [(SensitiveKind::Secret, hex)]
        );
        assert_eq!(
            found("https://api.example.com/v1/reset?token=a1B2c3D4e5F6g7H8&next=/home"),
            [(SensitiveKind::Secret, "a1B2c3D4e5F6g7H8")]
        );
    }

    #[test]
    fn ignores_hashes_versions_paths_and_identifiers() {
        for line in [
            "sha256: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            "https://github.com/CapSoftware/Cap/commit/3f9a2c4e8b7d6a5f4e3d2c1b0a9f8e7d6c5b4a39",
            "commit 3f9a2c4e8b7d6a5f4e3d2c1b0a9f8e7d6c5b4a39",
            "https://cdn.example.com/assets/app.3f9a2c4e8b7d6a5f.js?v=20240115",
            "550e8400-e29b-41d4-a716-446655440000",
            "/Users/aj/Library/Application Support/so.cap.desktop/recordings",
            "ThisIsAVeryLongIdentifierNameForAComponent2",
            "useSensitiveInformationDetectionAndBlurring",
            "version 1.2.3 released 2024-01-15 at 10:30",
            "rust-toolchain 1.88.0-aarch64-apple-darwin",
            "node_modules/@cap/web-domain/dist/index.d.ts",
        ] {
            assert!(found(line).is_empty(), "{line:?}: {:?}", found(line));
        }
    }

    #[test]
    fn finds_phone_numbers() {
        for (line, phone) in [
            ("phone +1 415-555-0132", "+1 415-555-0132"),
            ("call (415) 555-0132 now", "(415) 555-0132"),
            ("415.555.0132", "415.555.0132"),
            ("+44 20 7946 0958", "+44 20 7946 0958"),
            ("tel:+4915112345678", "+4915112345678"),
        ] {
            assert_eq!(found(line), [(SensitiveKind::Phone, phone)], "{line:?}");
        }
    }

    #[test]
    fn ignores_numbers_that_are_not_phones() {
        for line in [
            "released 2024-01-15",
            "ip 192.168.100.200",
            "version 10.15.7",
            "123-456-7890",
            "415-555-01325",
            "+0 123 456 789",
            "+3 points",
            "port 8080, pid 12345",
        ] {
            assert!(
                !kinds(line).contains(&SensitiveKind::Phone),
                "{line:?}: {:?}",
                found(line)
            );
        }
    }

    #[test]
    fn options_switch_detectors_off() {
        let line = "jane@example.com 4242 4242 4242 4242 +1 415-555-0132";
        let none = DetectorOptions {
            emails: false,
            cards: false,
            secrets: false,
            phones: false,
            faces: false,
        };
        assert!(find_sensitive_text(line, &none).is_empty());
        assert_eq!(
            find_sensitive_text(line, &DetectorOptions::default()).len(),
            3
        );
    }

    #[test]
    fn overlapping_matches_become_one_span() {
        let line = concat!("API_KEY=sk", "-proj-Xr7kQ2mN9vB4tL8wZ1cY6pH3");
        assert_eq!(
            found(line),
            [(
                SensitiveKind::ApiKey,
                concat!("sk", "-proj-Xr7kQ2mN9vB4tL8wZ1cY6pH3")
            )]
        );
    }

    #[test]
    fn handles_non_ascii_text() {
        let line = "Café — jane.doe@example.com • ✓ 4242 4242 4242 4242";
        assert_eq!(
            found(line),
            [
                (SensitiveKind::Email, "jane.doe@example.com"),
                (SensitiveKind::CardNumber, "4242 4242 4242 4242")
            ]
        );
    }

    type MakeSecret = Box<dyn FnMut(&mut Rng) -> String>;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn string(&mut self, alphabet: &[u8], len: usize) -> String {
            (0..len)
                .map(|_| alphabet[self.below(alphabet.len())] as char)
                .collect()
        }
    }

    const BASE64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const BASE64URL: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

    fn covered(line: &str, secret: &str) -> bool {
        let start = line.find(secret).unwrap();
        let end = start + secret.len();
        find_sensitive_text(line, &DetectorOptions::default())
            .iter()
            .any(|item| item.range.start <= start + 1 && item.range.end + 1 >= end)
    }

    fn coverage(samples: usize, mut make: impl FnMut(&mut Rng) -> String) -> f64 {
        let mut rng = Rng(7);
        let hits = (0..samples)
            .filter(|_| {
                let secret = make(&mut rng);
                covered(&secret, &secret)
            })
            .count();
        hits as f64 / samples as f64
    }

    #[test]
    fn random_secrets_are_found_without_any_label() {
        let cases: [(&str, MakeSecret, f64); 5] = [
            (
                "base64, 32-64 chars",
                Box::new(|rng: &mut Rng| {
                    let len = 32 + rng.below(33);
                    rng.string(BASE64, len)
                }),
                0.97,
            ),
            (
                "base64url, 32-64 chars",
                Box::new(|rng: &mut Rng| {
                    let len = 32 + rng.below(33);
                    rng.string(BASE64URL, len)
                }),
                0.97,
            ),
            (
                "base64, 40-64 chars",
                Box::new(|rng: &mut Rng| {
                    let len = 40 + rng.below(25);
                    rng.string(BASE64, len)
                }),
                0.995,
            ),
            (
                "PEM body line",
                Box::new(|rng: &mut Rng| rng.string(BASE64, 64)),
                0.999,
            ),
            (
                "PKCS#8 first line",
                Box::new(|rng: &mut Rng| {
                    format!(
                        "{}{}",
                        concat!("MIIEvQIBADANBgkqhkiG9w0B", "AQEFAASCBKcwggSjAgEAAoIBAQ"),
                        rng.string(BASE64, 14)
                    )
                }),
                1.0,
            ),
        ];
        for (name, make, floor) in cases {
            let rate = coverage(2000, make);
            eprintln!("{name}: {:.2}% found", rate * 100.0);
            assert!(rate >= floor, "{name}: {rate}");
        }
    }

    #[test]
    fn generated_identifiers_are_not_secrets() {
        const WORDS: &[&str] = &[
            "use",
            "sensitive",
            "information",
            "detection",
            "blur",
            "component",
            "provider",
            "context",
            "render",
            "editor",
            "timeline",
            "segment",
            "recording",
            "export",
            "upload",
            "settings",
            "handler",
            "factory",
            "manager",
            "service",
            "default",
        ];
        let mut rng = Rng(11);
        for _ in 0..2000 {
            let mut identifier = String::new();
            while identifier.len() < 40 {
                let word = WORDS[rng.below(WORDS.len())];
                identifier.push_str(&word[..1].to_ascii_uppercase());
                identifier.push_str(&word[1..]);
                if rng.below(4) == 0 {
                    identifier.push_str(&rng.below(100).to_string());
                }
            }
            assert!(found(&identifier).is_empty(), "{identifier}");
        }
    }

    #[test]
    fn private_key_blocks_are_blurred_line_by_line() {
        let lines = [
            concat!("-----BEGIN ", "PRIVATE KEY-----"),
            concat!("MIIEvQIBADANBgkqhkiG9w0B", "AQEFAASCBKcwggSjAgEAAoIBAQC7"),
            "abc/def+ghi",
            "Zm9v==",
            concat!("-----END ", "PRIVATE KEY-----"),
            "after the block",
        ];
        let found = find_sensitive_lines(&lines, &DetectorOptions::default());
        assert!(found[0].is_empty());
        let ocr_dashes = [
            concat!("—---BEGIN ", "PRIVATE KEY—---"),
            "Zm9v==",
            "-—--END",
        ];
        let read = find_sensitive_lines(&ocr_dashes, &DetectorOptions::default());
        assert_eq!(read[1].len(), 1);
        assert!(read[2].is_empty());
        for (line, items) in lines.iter().zip(&found).skip(1).take(3) {
            assert_eq!(items.len(), 1, "{line}");
            assert_eq!(items[0].range, 0..line.len());
        }
        assert!(found[4].is_empty() && found[5].is_empty());
    }

    #[test]
    fn finds_credentials_inside_urls() {
        for (line, secret) in [
            (
                "postgres://admin:s3cretPW99@localhost:5432/app",
                "s3cretPW99",
            ),
            ("redis://:Zx9pQ2rT7vL4@127.0.0.1:6379", "Zx9pQ2rT7vL4"),
            (
                "mongodb+srv://app:hunter2hunter2@cluster0.example.net/db",
                "hunter2hunter2",
            ),
            ("mysql://root:rootpass@db:3306/shop", "rootpass"),
            ("amqp://guest:guestpass@rabbit:5672/", "guestpass"),
            (
                concat!(
                    "https://hooks.slack.com/services/",
                    "T0001/B0002/XXXXabcdYYYYefgh1234"
                ),
                "T0001/B0002/XXXXabcdYYYYefgh1234",
            ),
            (
                concat!(
                    "https://discord.com/api/webhooks/",
                    "123456789012345678/abcDEF_ghiJKL-mnoPQR"
                ),
                "123456789012345678/abcDEF_ghiJKL-mnoPQR",
            ),
            (
                "https://storage.googleapis.com/b/o?X-Goog-Signature=4f3e2d1c0b9a8f7e6d5c&x=1",
                "4f3e2d1c0b9a8f7e6d5c",
            ),
            (
                "https://s3.amazonaws.com/b/k?X-Amz-Signature=9a8b7c6d5e4f3a2b1c0d",
                "9a8b7c6d5e4f3a2b1c0d",
            ),
        ] {
            let items = found(line);
            assert!(
                items.iter().any(|(_, text)| *text == secret),
                "{line:?}: {items:?}"
            );
        }
        let bot = concat!("12345678", "9:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw1");
        assert_eq!(
            found(&format!("https://api.telegram.org/bot{bot}/getMe")),
            [(SensitiveKind::ApiKey, bot)]
        );
        assert_eq!(
            found(&format!("TELEGRAM={bot}")),
            [(SensitiveKind::ApiKey, bot)]
        );
    }

    #[test]
    fn urls_without_credentials_stay_clear() {
        for line in [
            "postgres://localhost:5432/app",
            "https://user@example.com/path",
            "https://hooks.slack.com/",
            "https://discord.com/channels/123/456",
            "at 10:30:45 the job ran",
            "https://example.com/?q=search&page=2",
        ] {
            assert!(found(line).is_empty(), "{line:?}: {:?}", found(line));
        }
    }

    #[test]
    fn labeled_passwords_need_no_particular_shape() {
        for (line, value) in [
            ("password: correcthorsebattery", "correcthorsebattery"),
            ("PASSWORD=letmeinplease", "letmeinplease"),
            ("password: $ecret!Pass", "$ecret!Pass"),
            ("password: @dm1n", "@dm1n"),
            ("password: %Pa55%word", "%Pa55%word"),
            ("password: hunter(2)", "hunter(2"),
            ("password: p<ss", "p<ss"),
            ("token = 7f3k", "7f3k"),
            ("secret: Summer2024", "Summer2024"),
        ] {
            let items = found(line);
            assert!(
                items.iter().any(|(_, text)| *text == value),
                "{line:?}: {items:?}"
            );
        }
    }

    #[test]
    fn labeled_types_and_code_are_not_passwords() {
        for line in [
            "password: SecureString,",
            "password: PasswordHash,",
            "password: string",
            "password: str",
            "fn login(password: &str)",
            "password: Option<String>,",
            "password: Vec<u8>",
            "let password = getPassword();",
            "token = os.getenv(\"TOKEN\")",
            "password: $DB_PASSWORD",
            "password: %DB_PASSWORD%",
            "secret: ${{ secrets.DEPLOY_KEY }}",
            "max_token: 4096",
        ] {
            assert!(found(line).is_empty(), "{line:?}: {:?}", found(line));
        }
    }

    #[test]
    fn real_country_domains_are_addresses() {
        for address in [
            "milan@firma.rs",
            "juan@empresa.com.py",
            "ion@site.md",
            "a@shop.zip",
        ] {
            assert_eq!(found(address), [(SensitiveKind::Email, address)]);
        }
        for line in ["icon@2x.png", "logo@3X.svg", "hero@2x.rs"] {
            assert!(found(line).is_empty(), "{line:?}: {:?}", found(line));
        }
    }

    #[test]
    fn digests_data_uris_and_words_containing_key_are_clear() {
        let sri = concat!(
            "integrity: sha512-",
            "Zm9vYmFyQmF6UXV4MTIzNDU2Nzg5MGFiY2RlZkdISUpLTE1OT1BRUlNUVVZXWFla",
            "YWJjZGVmZ2hpams0NTY3ODk="
        );
        let data_uri = concat!(
            "src=\"data:image/png;base64,",
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==\""
        );
        for line in [
            sri,
            data_uri,
            "monkey 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            "keyboard 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        ] {
            assert!(found(line).is_empty(), "{line:?}: {:?}", found(line));
        }
        let hex = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";
        assert_eq!(
            found(&format!("apiKey {hex}")),
            [(SensitiveKind::Secret, hex)]
        );
    }
}
