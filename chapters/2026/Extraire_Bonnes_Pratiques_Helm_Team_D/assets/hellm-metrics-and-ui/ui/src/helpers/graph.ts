import { GraphData } from "force-graph";

export interface IGraph {
  name: string,
  list: Record<string, string[]>,
  types: Record<string, string>
}

export function convertToGraphData(graph: IGraph): GraphData {
  let graphData: GraphData = { nodes: [], links: [] };
  for (const node of Object.keys(graph.types)) {
    graphData.nodes.push({ id: node })
  }
  for (const [source, targets] of Object.entries(graph.list)) {
    for (const target of targets) {
      graphData.links.push({ source, target });
    }
  }
  return graphData
}
