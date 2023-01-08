import * as ts from "typescript";
import { factory } from "typescript";

export default function transformer(
  program: ts.Program
): ts.TransformerFactory<ts.Node> {
  return (context) => (node) => visitNodeAndChildren(node, program, context);
}

function visitNodeAndChildren(
  node: ts.Node,
  program: ts.Program,
  context: ts.TransformationContext
): ts.Node {
  if (ts.isCallExpression(node)) {
    return visitCallExpr(node, program);
  }
  return ts.visitEachChild(
    node,
    (child) => visitNodeAndChildren(child, program, context),
    context
  );
}

function visitCallExpr(node: ts.CallExpression, program: ts.Program): ts.Node {
  const checker = program.getTypeChecker();
  const signature = checker.getResolvedSignature(node);
  const name = checker.getTypeAtLocation(signature!.declaration!).symbol.name;
  if (name !== "__collectProperties") {
    return node;
  }
  const typeArg = node.typeArguments![0];
  const type = checker.getTypeFromTypeNode(typeArg);
  const collector = new PropertyCollector(node, checker);
  return factory.createObjectLiteralExpression(
    checker.getPropertiesOfType(type).map((prop) => collector.collect(prop)),
    true
  );
}

class PropertyCollector {
  constructor(
    private readonly node: ts.Node,
    private readonly checker: ts.TypeChecker
  ) {}

  collect(property: ts.Symbol): ts.PropertyAssignment {
    const type = this.checker.getTypeOfSymbolAtLocation(property, this.node);
    return factory.createPropertyAssignment(
      litStr(property.name),
      createObjectLiteral({
        className: this.collectClassName(
          this.getPropertyType(type, "className")
        ),
        props: this.collectProperties(this.getPropertyType(type, "props")),
        events: this.collectEvents(this.getPropertyType(type, "events")),
      })
    );
  }

  private getPropertyType(type: ts.Type, property: string): ts.Type {
    const sym = this.checker.getPropertyOfType(type, property)!;
    return this.checker
      .getTypeOfSymbolAtLocation(sym, this.node)
      .getNonNullableType();
  }

  private collectEvents(type: ts.Type): ts.ObjectLiteralExpression {
    const checker = this.checker;
    const sig = type.getCallSignatures();
    if (sig.length === 0) {
      return factory.createObjectLiteralExpression([]);
    }
    const k = sig[0].typeParameters![0].symbol
      .declarations![0] as ts.TypeParameterDeclaration;
    const eventMap = checker.getTypeFromTypeNode(
      (k.constraint! as ts.TypeOperatorNode).type
    );
    const events = eventMap.getProperties().map((sym) => {
      const name = sym.getName();
      const type = checker
        .getTypeOfSymbolAtLocation(sym, this.node)
        .symbol.getName();
      return factory.createPropertyAssignment(litStr(name), litStr(type));
    });
    return factory.createObjectLiteralExpression(events, true);
  }

  private collectClassName(type: ts.Type): ts.StringLiteral {
    return litStr(type.symbol?.name || "undefined");
  }

  private collectProperties(type: ts.Type): ts.ObjectLiteralExpression {
    const collected = type
      .getProperties()
      .map((sym) => this.collectProperty(sym))
      .filter((t): t is ts.PropertyAssignment => t !== undefined);
    return factory.createObjectLiteralExpression(collected, true);
  }

  private collectProperty(
    property: ts.Symbol
  ): ts.PropertyAssignment | undefined {
    const declaration = property.declarations![0];
    const modifier = ts.getCombinedModifierFlags(declaration);
    if (modifier & (ts.ModifierFlags.Readonly | ts.ModifierFlags.Deprecated)) {
      return;
    }

    const type = this.checker
      .getTypeOfSymbolAtLocation(property, this.node)
      .getNonNullableType();
    let value: ts.Expression | undefined;
    if (type.flags & ts.TypeFlags.Boolean) {
      value = litStr("boolean");
    } else if (type.flags & ts.TypeFlags.Number) {
      value = litStr("number");
    } else if (type.flags & ts.TypeFlags.String) {
      value = litStr("string");
    } else if (type.isUnion()) {
      value = this.collectLiteralProperty(type);
    }
    if (value === undefined) {
      return;
    }
    return factory.createPropertyAssignment(litStr(property.name), value);
  }

  private collectLiteralProperty(
    type: ts.UnionType
  ): ts.Expression | undefined {
    const literals = new Set<string>();
    for (const ty of type.types) {
      if (ty.isStringLiteral()) {
        literals.add(ty.value);
      } else if (ty.flags & ts.TypeFlags.BooleanLiteral) {
        literals.add((ty as any).intrinsicName);
      }
    }
    if (literals.size === 0) {
      let jsType = "string";
      for (const ty of type.types) {
        if (ty.flags & ts.TypeFlags.Number) {
          jsType = "number";
        } else if (ty.flags & ts.TypeFlags.Boolean) {
          jsType = "boolean";
        } else if (!(ty.flags & ts.TypeFlags.String)) {
          return;
        }
      }
      return litStr(jsType);
    }
    literals.delete("");
    const values = Array.from(literals.values()).sort();
    if (arrayEqual(values, ["false", "true"])) {
      return litStr("boolean");
    }
    return factory.createArrayLiteralExpression(values.map(litStr));
  }
}

function arrayEqual<T>(a: T[], b: T[]): boolean {
  if (a === b) {
    return true;
  } else if (a.length !== b.length) {
    return false;
  } else {
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) {
        return false;
      }
    }
    return true;
  }
}

function createObjectLiteral(obj: {
  [k: string]: ts.Expression;
}): ts.ObjectLiteralExpression {
  return factory.createObjectLiteralExpression(
    Object.entries(obj).map(([key, val]) =>
      factory.createPropertyAssignment(litStr(key), val)
    ),
    true
  );
}

function litStr(s: string): ts.StringLiteral {
  return factory.createStringLiteral(s);
}
