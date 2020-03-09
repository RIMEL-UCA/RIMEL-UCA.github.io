package fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.neo4j.ogm.annotation.GeneratedValue;
import org.neo4j.ogm.annotation.Id;
import org.neo4j.ogm.annotation.NodeEntity;
import org.neo4j.ogm.annotation.Relationship;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@NodeEntity
@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class MavenProfile implements Serializable {

    @Id
    @GeneratedValue
    private Long id;

    private String name;

    @Builder.Default
    @Relationship
    private List<HasProperty> properties = new ArrayList<>();

    @Builder.Default
    @Relationship
    private List<HasPlugin> plugins = new ArrayList<>();

    @Builder.Default
    @Relationship
    private List<HasDependency> dependencies = new ArrayList<>();

    public void addDependency(final HasDependency dependency) {
        this.dependencies.add(dependency);
    }

    public void addProperty(final HasProperty property) {
        this.properties.add(property);
    }

    public void addPlugin(final HasPlugin plugin) {
        this.plugins.add(plugin);
    }
}
