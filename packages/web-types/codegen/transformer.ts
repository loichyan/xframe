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
      property.name,
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
    const k = type.getCallSignatures()[0].typeParameters![0].symbol
      .declarations![0] as ts.TypeParameterDeclaration;
    const eventMap = checker.getTypeFromTypeNode(
      (k.constraint! as ts.TypeOperatorNode).type
    );
    const events = eventMap.getProperties().map((sym) => {
      const name = sym.getName();
      const type = checker
        .getTypeOfSymbolAtLocation(sym, this.node)
        .symbol.getName();
      return factory.createPropertyAssignment(
        name,
        factory.createStringLiteral(type)
      );
    });
    return factory.createObjectLiteralExpression(events, true);
  }

  private collectClassName(type: ts.Type): ts.StringLiteral {
    return factory.createStringLiteral(type.symbol.name);
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

    let value: ts.Expression;
    const type = this.checker
      .getTypeOfSymbolAtLocation(property, this.node)
      .getNonNullableType();
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
    return factory.createPropertyAssignment(property.name, value);
  }

  private collectLiteralProperty(
    type: ts.UnionType
  ): ts.ArrayLiteralExpression | undefined {
    const literals = [];
    for (const lit of type.types) {
      if (lit.isStringLiteral()) {
        literals.push(lit.value);
      }
    }
    if (literals.length === 0) {
      return;
    }
    return factory.createArrayLiteralExpression(
      literals.map((lit) => factory.createStringLiteral(lit))
    );
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
