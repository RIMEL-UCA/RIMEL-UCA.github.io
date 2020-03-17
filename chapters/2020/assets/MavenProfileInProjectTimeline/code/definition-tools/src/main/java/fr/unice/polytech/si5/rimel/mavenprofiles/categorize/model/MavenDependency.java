package fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.neo4j.ogm.annotation.GeneratedValue;
import org.neo4j.ogm.annotation.Id;
import org.neo4j.ogm.annotation.NodeEntity;

import java.util.ArrayList;
import java.util.List;

@NodeEntity
@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class MavenDependency {

	@Id
	@GeneratedValue
	private Long id;

	private String artifactId;
}
