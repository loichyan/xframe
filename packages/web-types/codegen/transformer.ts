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
    const collected = [];
    for (const prop of this.checker.getPropertiesOfType(
      this.getPropertyType(type, "props")
    )) {
      const collectedProp = this.collectProperty(prop);
      if (collectedProp !== undefined) {
        collected.push(collectedProp);
      }
    }
    return factory.createPropertyAssignment(
      property.name,
      createObjectLiteral({
        className: factory.createStringLiteral(
          this.getPropertyType(type, "className").symbol.name
        ),
        props: factory.createObjectLiteralExpression(collected, true),
      })
    );
  }

  private getPropertyType(type: ts.Type, property: string): ts.Type {
    const sym = this.checker.getPropertyOfType(type, property)!;
    return this.checker.getTypeOfSymbolAtLocation(sym, this.node);
  }

  private collectProperty(
    property: ts.Symbol
  ): ts.PropertyAssignment | undefined {
    const declaration = property.declarations![0];
    const modifier = ts.getCombinedModifierFlags(declaration);
    if (modifier & (ts.ModifierFlags.Readonly | ts.ModifierFlags.Deprecated)) {
      return;
    }

    let value: ts.Expression;
    const type = this.checker
      .getTypeOfSymbolAtLocation(property, this.node)
      .getNonNullableType();
    if (property.name.startsWith("on")) {
      const def = type.symbol.declarations![0];
      if (!ts.isFunctionTypeNode(def)) {
        return;
      }
      value = this.collectEventProperty(def);
    } else {
      if (type.flags & ts.TypeFlags.Boolean) {
        value = factory.createStringLiteral("boolean");
      } else if (type.flags & ts.TypeFlags.Number) {
        value = factory.createStringLiteral("number");
      } else if (type.flags & ts.TypeFlags.String) {
        value = factory.createStringLiteral("string");
      } else if (type.isUnion()) {
        const lit = this.collectLiteralProperty(type);
        if (lit === undefined) {
          return;
        }
        value = lit;
      } else {
        return;
      }
    }
    return factory.createPropertyAssignment(property.name, value);
  }

  private collectEventProperty(type: ts.FunctionTypeNode): ts.Expression {
    const eventType = type.parameters[1].type!.getText();
    return createObjectLiteral({
      type: factory.createStringLiteral("event"),
      eventType: factory.createStringLiteral(eventType),
    });
  }

  private collectLiteralProperty(
    type: ts.UnionType
  ): ts.Expression | undefined {
    const literals = [];
    for (const lit of type.types) {
      if (lit.isStringLiteral()) {
        literals.push(lit.value);
      }
    }
    if (literals.length === 0) {
      return;
    }
    return createObjectLiteral({
      type: factory.createStringLiteral("literals"),
      values: factory.createArrayLiteralExpression(
        literals.map((lit) => factory.createStringLiteral(lit))
      ),
    });
  }
}

function createObjectLiteral(obj: {
  [k: string]: ts.Expression;
}): ts.ObjectLiteralExpression {
  return factory.createObjectLiteralExpression(
    Object.entries(obj).map(([key, val]) =>
      factory.createPropertyAssignment(key, val)
    ),
    true
  );
}
