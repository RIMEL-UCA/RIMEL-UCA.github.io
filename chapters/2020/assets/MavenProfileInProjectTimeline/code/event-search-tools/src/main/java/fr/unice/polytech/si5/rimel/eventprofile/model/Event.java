package fr.unice.polytech.si5.rimel.eventprofile.model;

import fr.unice.polytech.si5.rimel.eventprofile.domain.DiffType;
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
public class Event implements Serializable {

    @Id
    @GeneratedValue
    private Long id;

    private DiffType diffType;

    @Builder.Default
    private boolean isMerge = false;

    private String commitMsg;

    private String tag;

    private List<String> branches;
}
