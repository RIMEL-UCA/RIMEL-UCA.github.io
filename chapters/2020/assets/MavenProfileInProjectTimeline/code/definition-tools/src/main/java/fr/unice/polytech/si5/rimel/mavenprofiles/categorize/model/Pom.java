package fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.neo4j.ogm.annotation.*;
import org.springframework.stereotype.Component;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import static org.neo4j.ogm.annotation.Relationship.OUTGOING;


@NodeEntity
@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Component
public class Pom implements Serializable {

    @Id
    @GeneratedValue
    private Long id;

    private String fullRepositoryName;

    private String path;

    private int size;

    @Builder.Default
    @Relationship
    private List<HasProfile> profiles = new ArrayList<>();

    private Orm orm;

    public void addProfile(final HasProfile profile) {
        this.profiles.add(profile);
    }

}
