package fr.unice.polytech.si5.rimel.eventprofile.model;

import lombok.*;
import org.neo4j.ogm.annotation.GeneratedValue;
import org.neo4j.ogm.annotation.Id;
import org.neo4j.ogm.annotation.NodeEntity;
import org.neo4j.ogm.annotation.Relationship;
import org.springframework.stereotype.Component;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;


@NodeEntity
@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
@EqualsAndHashCode
@Component
public class Pom implements Serializable {

    @Id
    @GeneratedValue
    private Long id;

    private String fullRepositoryName;

    private String path;

    @Builder.Default
    @Relationship
    private List<HasProfile> profiles = new ArrayList<>();

    //private Orm orm;

    @Relationship
    @Builder.Default
    private List<Event> events = new ArrayList<>();

    public void addProfile(final HasProfile profile) {
        this.profiles.add(profile);
    }

    public void addEvent(final Event event) {
        this.events.add(event);
    }

}
