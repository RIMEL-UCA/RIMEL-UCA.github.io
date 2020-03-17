package fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.neo4j.ogm.annotation.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.neo4j.ogm.annotation.Relationship.INCOMING;

@NodeEntity
@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class MavenProperty {

	@Id
	@GeneratedValue
	private Long id;

	private String name;

	@Builder.Default
	private List<String> values = new ArrayList<>();

}
