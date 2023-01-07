use crate::{
    lexer::{Lexer, Token},
    Element, Event, JsType, Literals, Property,
};
use std::collections::BTreeMap;

type Result<T> = std::result::Result<T, String>;
type Predefined<'a> = BTreeMap<&'a str, Content<'a>>;
type Elements<'a> = Vec<Element<'a>>;

#[derive(Default)]
struct Content<'a> {
    attributes: Vec<Property<'a>>,
    events: Vec<Event<'a>>,
}

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    predefined: Predefined<'a>,
    elements: Elements<'a>,
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            lexer: Lexer::new(source),
            predefined: Default::default(),
            elements: Default::default(),
        }
    }

    pub fn parse(mut self) -> Result<Elements<'a>> {
        if let Err(e) = self.parse_impl() {
            let (line, char) = self.lexer.pos();
            Err(format!("{e} at line {line} character {char}",))
        } else {
            Ok(self.elements)
        }
    }

    fn parse_impl(&mut self) -> Result<()> {
        while let Some(token) = self.lexer.next() {
            match token {
                // A predefined preset.
                Token::Char(b'+') => {
                    self.parse_predefined()?;
                }
                Token::Ident => {
                    let name = self.lexer.buf();
                    let preset = self.parse_preset(name)?;
                    self.elements.push(preset);
                }
                _ => todo!(),
            }
        }
        Ok(())
    }

    fn parse_predefined(&mut self) -> Result<()> {
        let name = self.lexer.expect_ident()?;
        let content = self.parse_content()?;
        self.predefined.insert(name, content);
        Ok(())
    }

    fn parse_preset(&mut self, name: &'a str) -> Result<Element<'a>> {
        self.lexer.expect_char(b'=')?;
        self.lexer.expect_char(b'>')?;
        let js_class = self.lexer.expect_ident()?;
        let content = self.parse_content()?;
        Ok(Element {
            name,
            js_class,
            attributes: content.attributes,
            events: content.events,
        })
    }

    fn parse_content(&mut self) -> Result<Content<'a>> {
        self.lexer.expect_char(b'{')?;
        ParseContent {
            predefined: &self.predefined,
            lexer: &mut self.lexer,
            content: Default::default(),
        }
        .parse()
    }
}

impl<'a> Lexer<'a> {
    fn expect_ident(&mut self) -> Result<&'a str> {
        if let Some(Token::Ident) = self.next() {
            Ok(self.buf())
        } else {
            Err("expect an identifier".to_owned())
        }
    }

    fn expect_char(&mut self, ch: u8) -> Result<u8> {
        match self.next() {
            Some(Token::Char(got)) if got == ch => Ok(got),
            _ => Err(format!("expect '{}'", ch as char)),
        }
    }
}

struct ParseContent<'a, 'b> {
    predefined: &'b Predefined<'a>,
    lexer: &'b mut Lexer<'a>,
    content: Content<'a>,
}

impl<'a> ParseContent<'a, '_> {
    fn parse(mut self) -> Result<Content<'a>> {
        while let Some(token) = self.lexer.next() {
            match token {
                Token::Char(b'}') => break,
                // Extends a predefined.
                Token::Char(b'+') => {
                    let name = self.lexer.expect_ident()?;
                    let predefined = self
                        .predefined
                        .get(name)
                        .ok_or_else(|| format!("unknown predefined '{}'", name))?;
                    self.content
                        .attributes
                        .extend(predefined.attributes.iter().cloned());
                    self.content
                        .events
                        .extend(predefined.events.iter().cloned());
                }
                // A event.
                Token::Char(b'@') => {
                    let name = self.lexer.expect_ident()?;
                    self.lexer.expect_char(b':')?;
                    let js_class = self.lexer.expect_ident()?;
                    self.content.events.push(Event { name, js_class })
                }
                Token::Char(b'*') => {
                    let name = self.lexer.expect_ident()?;
                    let js_type = self.parse_js_type()?;
                    self.content.attributes.push(Property {
                        name,
                        js_type,
                        attribute: true,
                    })
                }
                // An attribute.
                Token::Ident => {
                    let name = self.lexer.buf();
                    let js_type = self.parse_js_type()?;
                    self.content.attributes.push(Property {
                        name,
                        js_type,
                        attribute: false,
                    });
                }
                _ => return Err("unknown token".to_owned()),
            }
            self.lexer.expect_char(b',')?;
        }
        Ok(self.content)
    }

    fn parse_js_type(&mut self) -> Result<JsType<'a>> {
        self.lexer.expect_char(b':')?;
        Ok(if let Some(Token::Char(b'(')) = self.lexer.peek() {
            self.lexer.next();
            let mut values = Vec::new();
            while let Some(token) = self.lexer.next() {
                match token {
                    Token::Char(b')') => break,
                    Token::String => values.push(self.lexer.buf()),
                    _ => return Err("unknown token".to_owned()),
                }
            }
            JsType::Literals(Literals { values })
        } else {
            JsType::Type(self.lexer.expect_ident()?)
        })
    }
}
