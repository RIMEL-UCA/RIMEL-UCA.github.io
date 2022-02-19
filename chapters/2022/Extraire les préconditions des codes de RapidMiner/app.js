
const MIN = 2
const MEDIUM = 4
const MAX = 6
let nodes = null;
let edges = null;
let network = null;
let arrowToTyp = {
    to: {
      enabled: true,
      type: "inv_curve",
    },
  }

function dictGraphToNodes(dictGraph){
    let nodes= []
    for (let elem of dictGraph){
        let cpt = 0
        for (let op of dictGraph[elem]){
            if(op == elem){
                cpt=cpt+1
            }
        }
        nodes.push({ id: elem, value: cpt, label:elem })
    }
    return nodes
}

function dictGraphToEdges(dictGraph){
    let edges= []
    for (let elem of dictGraph){
        for (let op of dictGraph[elem]){
            edges.push({ from: elem, to: op, value: 3, title: "3 emails per week",arrows: arrowToTyp })
        }
    }
    return edges
}

function draw() {
  // create people.
  // value corresponds with the age of the person
  nodes = [
    { id: "1", value: 2, label: "Algie" },
    { id: 2, value: 31, label: "Alston" },
    { id: 3, value: 12, label: "Barney" },
    { id: 4, value: 16, label: "Coley" },
    { id: 5, value: 17, label: "Grant" },
    { id: 6, value: 15, label: "Langdon" },
    { id: 7, value: 6, label: "Lee" },
    { id: 8, value: 5, label: "Merlin" },
    { id: 9, value: 30, label: "Mick" },
    { id: 10, value: 18, label: "Tod" },
  ];

  // create connections between people
  // value corresponds with the amount of contact between two people
  edges = [
    { from: "2", to: 8, value: 3, title: "3 emails per week",arrows: arrowToTyp },
    { from: 8, to: "2", value: 3, title: "3 emails per week",arrows: arrowToTyp },
    { from: 4, to: 6, value: 8, title: "8 emails per week" ,arrows: arrowToTyp},
    { from: 5, to: 7, value: 2, title: "2 emails per week" ,arrows: arrowToTyp},
    { from: 4, to: 5, value: "1", title: "1 emails per week" ,arrows: arrowToTyp},
    { from: "2", to: 3, value: 6, title: "6 emails per week" ,arrows: arrowToTyp},
    { from: 5, to: 3, value: 1, title: "1 emails per week" ,arrows: arrowToTyp},
  ];

  // Instantiate our network object.
  let container = document.getElementById("mynetwork");
  let data = {
    nodes: nodes,
    edges: edges,
  };
  let options = {
    nodes: {
      shape: "dot",
      scaling: {
        label: {
          min: 8,
          max: 20,
        },
      },
    },
  };
  network = new vis.Network(container, data, options);
}

window.addEventListener("load", () => {
  draw();
});
