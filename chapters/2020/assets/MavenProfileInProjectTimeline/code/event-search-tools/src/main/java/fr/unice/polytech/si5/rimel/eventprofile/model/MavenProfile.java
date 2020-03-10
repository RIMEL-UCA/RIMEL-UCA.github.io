package fr.unice.polytech.si5.rimel.eventprofile.model;

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

}
