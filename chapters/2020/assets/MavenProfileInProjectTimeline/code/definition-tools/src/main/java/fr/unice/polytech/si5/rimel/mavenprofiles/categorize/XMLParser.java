package fr.unice.polytech.si5.rimel.mavenprofiles.categorize;

import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.ls.DOMImplementationLS;
import org.w3c.dom.ls.LSSerializer;
import org.xml.sax.InputSource;

import javax.xml.parsers.DocumentBuilderFactory;
import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class XMLParser {
    private String xmlString;
    private LSSerializer serializer;
    private DOMImplementationLS ls;

    public XMLParser(String xmlString) {
        this.xmlString = xmlString;
    }

    public List<Node> parseNode(String tag) throws Exception {
        Document xmlDocument = DocumentBuilderFactory.newInstance()
                .newDocumentBuilder()
                .parse(new InputSource(new ByteArrayInputStream(xmlString.getBytes(StandardCharsets.UTF_8))));

        ls = (DOMImplementationLS) xmlDocument.getImplementation();
        serializer = ls.createLSSerializer();

        List<Node> nodeList = new ArrayList<>();

        searchTagInChildNode(xmlDocument.getFirstChild(), nodeList, tag);

        return nodeList;
    }

    void searchTagInChildNode(Node node, List<Node> nodeList, String tag) {
        if (node.getNodeName().equals(tag)) {
            nodeList.add(node);
            return;
        }

        if (node.hasChildNodes()) {
            NodeList elementsList = node.getChildNodes();
            for (int i = 0; i < elementsList.getLength(); i++) {
                searchTagInChildNode(elementsList.item(i), nodeList, tag);
            }
        }
    }

    public LSSerializer getSerializer() {
        return serializer;
    }

    public DOMImplementationLS getLs() {
        return ls;
    }
}
