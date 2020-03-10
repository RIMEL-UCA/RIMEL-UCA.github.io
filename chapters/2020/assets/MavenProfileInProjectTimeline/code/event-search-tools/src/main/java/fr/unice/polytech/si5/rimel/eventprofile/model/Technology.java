package fr.unice.polytech.si5.rimel.eventprofile.model;

import fr.unice.polytech.si5.rimel.eventprofile.domain.StackType;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.neo4j.ogm.annotation.GeneratedValue;
import org.neo4j.ogm.annotation.Id;
import org.neo4j.ogm.annotation.NodeEntity;

import java.io.Serializable;

@NodeEntity
@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class Technology implements Serializable {

    @Id
    @GeneratedValue
    private Long id;

    private StackType stack;
}
