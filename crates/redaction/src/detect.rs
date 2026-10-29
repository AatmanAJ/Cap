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
        find_labeled_secrets(&chars, &mut found);
        find_high_entropy(&chars, &mut found);
    }
    if options.phones {
        find_phones(&chars, &mut found);
    }
    merge_overlaps(line, found)
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

// -- Email ------------------------------------------------------------------

const FILE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "svg", "webp", "avif", "heic", "ico", "bmp", "tiff", "js", "mjs",
    "cjs", "ts", "tsx", "jsx", "css", "scss", "html", "json", "md", "txt", "rs", "py", "go", "rb",
    "java", "kt", "swift", "mp4", "mov", "webm", "mp3", "wav", "pdf", "zip", "lock", "toml",
    "yaml", "yml", "xml", "wasm", "map", "gz",
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
        if valid_email(local, domain) {
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
    (2..=24).contains(&tld.len())
        && tld.chars().all(|c| c.is_ascii_alphabetic())
        && !FILE_EXTENSIONS
            .iter()
            .any(|extension| tld.eq_ignore_ascii_case(extension))
}

// -- Payment cards ----------------------------------------------------------

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

// -- API keys and tokens ----------------------------------------------------

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

const PLACEHOLDERS: &[&str] = &[
    "string",
    "number",
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
            if looks_like_secret_value(value, label.starts_with("pass")) {
                chars.push(found, SensitiveKind::Secret, cursor..end, value.to_string());
            }
        }
    }
}

fn looks_like_secret_value(value: &str, password: bool) -> bool {
    let min_len = if password { 6 } else { 8 };
    if value.chars().count() < min_len || value.len() > 512 {
        return false;
    }
    if value.starts_with(['$', '{', '<', '(', '[', '%', '@'])
        || value.contains("process.env")
        || value.contains("environ")
        || value.contains("://")
        || value.contains(['(', '<'])
        || value.contains("::")
    {
        return false;
    }
    let lower = value.to_ascii_lowercase();
    if PLACEHOLDERS.contains(&lower.as_str())
        || lower.starts_with("your")
        || lower
            .chars()
            .all(|c| matches!(c, 'x' | '*' | '•' | '.' | '-' | '_'))
    {
        return false;
    }
    if value
        .chars()
        .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
        && value.contains('_')
    {
        return false;
    }
    let looks_like_path = value.split('.').count() > 1
        && value.split('.').all(|part| {
            !part.is_empty() && part.chars().all(|c| c.is_ascii_alphabetic() || c == '_')
        });
    if looks_like_path {
        return false;
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
    classes >= 2 && (password || has_letter_and_digit(value))
}

fn has_letter_and_digit(value: &str) -> bool {
    value.chars().any(|c| c.is_ascii_alphabetic()) && value.chars().any(|c| c.is_ascii_digit())
}

const CONTEXT_WORDS: &[&str] = &[
    "secret",
    "token",
    "key",
    "password",
    "passwd",
    "auth",
    "bearer",
    "credential",
    "signature",
    "sig=",
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
    "password",
    "auth",
    "code",
];

fn is_b64(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '_' | '-')
}

/// Long random-looking strings: base64 blobs and, only when the line says
/// it is a credential, hex. Plain hashes (commit SHAs, digests, asset
/// fingerprints) are everywhere on a developer's screen and are not secret,
/// so a bare hex run never counts and a URL only counts through a
/// credential-named query parameter.
fn find_high_entropy(chars: &Chars, found: &mut Vec<TextMatch>) {
    let lower = chars.line.to_ascii_lowercase();
    let mut index = 0;
    while index < chars.len() {
        if chars.get(index).is_some_and(char::is_whitespace) {
            index += 1;
            continue;
        }
        let word_end = chars.scan_while(index, |c| !c.is_whitespace());
        let word = index..word_end;
        let text = chars.text(word.clone());
        if text.contains("://") || text.starts_with("www.") {
            find_url_secrets(chars, word.clone(), found);
        } else {
            let context_before = |at: usize| {
                let before = &lower[..chars.byte(at)];
                CONTEXT_WORDS.iter().any(|word| before.contains(word))
            };
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
                if random_looking(candidate, context_before(start)) {
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

fn find_url_secrets(chars: &Chars, word: Range<usize>, found: &mut Vec<TextMatch>) {
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

fn random_looking(candidate: &str, in_context: bool) -> bool {
    let len = candidate.len();
    let is_hex = candidate.chars().all(|c| c.is_ascii_hexdigit() || c == '-');
    if is_hex {
        let digits = candidate.chars().filter(char::is_ascii_hexdigit).count();
        return in_context
            && digits >= 32
            && candidate.chars().any(|c| c.is_ascii_digit())
            && candidate.chars().any(|c| c.is_ascii_alphabetic());
    }
    if candidate.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    if candidate.matches('/').count() >= 2 && !candidate.contains(['+', '=']) {
        return false;
    }
    let has_lower = candidate.chars().any(|c| c.is_ascii_lowercase());
    let has_upper = candidate.chars().any(|c| c.is_ascii_uppercase());
    let digits = candidate.chars().filter(char::is_ascii_digit).count();
    let jagged = longest_run(candidate, |c| c.is_ascii_lowercase()) <= 6
        && longest_run(candidate, |c| c.is_ascii_uppercase()) <= 6;
    let entropy = shannon_entropy(candidate);
    if in_context {
        len >= 20 && has_letter_and_digit(candidate) && entropy >= 3.5 && jagged
    } else {
        len >= 32 && has_lower && has_upper && digits >= 2 && entropy >= 4.2 && jagged
    }
}

// -- Phone numbers ----------------------------------------------------------

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
}
