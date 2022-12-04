pub struct Lexer<'a> {
    source: &'a str,
    bytes: &'a [u8],
    start: usize,
    end: usize,
    cursor: usize,
    line: usize,
    char: usize,
    peeked: Option<Token>,
}

macro_rules! ident_start {
    () => {
        b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'-'
    };
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            bytes: source.as_bytes(),
            start: 0,
            end: 0,
            cursor: 0,
            line: 1,
            char: 1,
            peeked: None,
        }
    }

    pub fn buf(&self) -> &'a str {
        self.source
            .get(self.start..self.end)
            .unwrap_or_else(|| unreachable!())
    }

    pub fn pos(&self) -> (usize, usize) {
        (self.line, self.char)
    }

    pub fn peek(&mut self) -> Option<Token> {
        if self.peeked.is_none() {
            self.peeked = self.parse_next();
        }
        self.peeked
    }

    pub fn next(&mut self) -> Option<Token> {
        self.peeked.take().or_else(|| self.parse_next())
    }

    fn parse_next(&mut self) -> Option<Token> {
        loop {
            self.buf_start();
            if let Some(ch) = self.next_ch() {
                let token = match ch {
                    // Skip whitespaces
                    b' ' => continue,
                    b'\n' => {
                        self.new_line();
                        continue;
                    }
                    b'/' => {
                        // Skip comment
                        if let Some(b'/') = self.peek_ch() {
                            self.eat_line();
                            continue;
                        } else {
                            Token::Char(b'/')
                        }
                    }
                    b'"' => self.next_str(),
                    ident_start!() => self.next_ident(),
                    _ => {
                        self.buf_end();
                        Token::Char(ch)
                    }
                };
                return Some(token);
            } else {
                break;
            }
        }
        None
    }

    fn next_str(&mut self) -> Token {
        self.buf_start();
        while let Some(ch) = self.peek_ch() {
            if ch == b'"' {
                self.buf_end();
                self.advance();
                break;
            }
            self.advance();
        }
        Token::String
    }

    fn next_ident(&mut self) -> Token {
        while let Some(ident_start!() | b'0'..=b'9') = self.peek_ch() {
            self.advance();
        }
        self.buf_end();
        Token::Ident
    }

    fn eat_line(&mut self) {
        while let Some(ch) = self.next_ch() {
            if ch == b'\n' {
                self.new_line();
                return;
            }
        }
    }

    fn buf_start(&mut self) {
        self.start = self.cursor;
    }

    fn buf_end(&mut self) {
        self.end = self.cursor;
    }

    fn next_ch(&mut self) -> Option<u8> {
        if let Some(ch) = self.peek_ch() {
            self.advance();
            Some(ch)
        } else {
            None
        }
    }

    fn peek_ch(&self) -> Option<u8> {
        self.bytes.get(self.cursor).copied()
    }

    fn new_line(&mut self) {
        self.line += 1;
        self.char = 1;
    }

    fn advance(&mut self) {
        self.char += 1;
        self.cursor += 1;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Token {
    Ident,
    String,
    Char(u8),
}

#[test]
fn lexer() {
    let mut lexer = Lexer::new(
        r#"button => HtmlButtonElement {
            // Comments ignored.
            aria-disabled: boolean,
            type: "button",
        }"#,
    );
    for expected in [
        "button",
        "=",
        ">",
        "HtmlButtonElement",
        "{",
        "aria-disabled",
        ":",
        "boolean",
        ",",
        "type",
        ":",
        "button",
        ",",
        "}",
    ] {
        lexer.next();
        assert_eq!(lexer.buf(), expected);
    }
}
