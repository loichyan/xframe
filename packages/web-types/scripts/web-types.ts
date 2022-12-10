import * as fs from "fs";

interface TypeLiterals {
  type: "literals";
  values: string[];
}

interface TypeEventHandler {
  type: "event";
  eventType: string;
}

type PropertyType =
  | "string"
  | "number"
  | "boolean"
  | TypeLiterals
  | TypeEventHandler;

type Properties = { [k: string]: PropertyType };

declare function __collectProperties<
  T extends { [k: string]: { className: any; props: any } }
>(): {
  [K in keyof T]: { className: string; props: Properties };
};

const HTML_PROPS = __collectProperties<{
  html: {
    className: HTMLElement;
    props: Omit<HTMLElement, keyof Node>;
  };
}>().html.props;

const ELEMENT_PROPS_MAP = __collectProperties<{
  [K in keyof HTMLElementTagNameMap]: {
    className: HTMLElementTagNameMap[K];
    props: Omit<HTMLElementTagNameMap[K], keyof HTMLElement>;
  };
}>();

function toPasalCase(s: string): string {
  return `${s[0].toUpperCase()}${s.slice(1)}`;
}

class PresetCollector {
  private result = "";

  collect(): string {
    this.result = "// This file is @generated by web-types.\n";
    // Collect properties of HTMLElement.
    this.startInsertProp("+HTML");
    this.collectProps(HTML_PROPS);
    this.endInsertProp();

    // Collect properties of each Element.
    for (const [tag, { className, props }] of Object.entries(
      ELEMENT_PROPS_MAP
    )) {
      this.startInsertProp(`${tag} => ${className}`);
      this.insertProp(`+HTML`);
      this.collectProps(props, tag);
      this.endInsertProp();
    }
    return this.result;
  }

  private collectProps(props: Properties, tag?: string) {
    const litTypePrefix = tag ? toPasalCase(tag) : "";
    for (const [key, val] of Object.entries(props)) {
      let name = key;
      let type: string;
      if (typeof val === "string") {
        type = val;
      } else {
        switch (val.type) {
          case "literals":
            type = this.literalType(`${litTypePrefix}${toPasalCase(key)}`, val);
            break;
          case "event":
            name = `@${key.slice(2)}`;
            type = val.eventType;
            break;
        }
      }
      this.insertProp(`${name}: ${type}`);
    }
  }

  private literalType(name: string, type: TypeLiterals) {
    let values = type.values.map((v) => `"${v}"`).join(" ");
    let result = `${name}(${values})`;
    return result;
  }

  private startInsertProp(head: string) {
    this.result += `${head} {`;
  }

  private insertProp(content: string) {
    this.result += `\n    ${content},`;
  }

  private endInsertProp() {
    this.result += `\n}\n`;
  }
}

const path = process.argv[2];
const preset = new PresetCollector().collect();
if (path === undefined) {
  console.log(preset);
} else {
  fs.writeFileSync(path, preset);
}
