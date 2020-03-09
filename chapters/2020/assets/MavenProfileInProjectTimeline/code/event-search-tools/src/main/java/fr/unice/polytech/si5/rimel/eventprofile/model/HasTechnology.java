package fr.unice.polytech.si5.rimel.eventprofile.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import org.neo4j.ogm.annotation.*;

@RelationshipEntity
@Builder
@AllArgsConstructor
public class HasTechnology {

	@Id
	@GeneratedValue
	private Long id;

	private String name;

	@StartNode
	private Pom pom;

	@EndNode
	private Technology technology;

	public HasTechnology() {
	}

	public Long getId() {
		return id;
	}

	public void setId(Long id) {
		this.id = id;
	}

	public Pom getPom() {
		return pom;
	}

	public void setPom(Pom pom) {
		this.pom = pom;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}
}
