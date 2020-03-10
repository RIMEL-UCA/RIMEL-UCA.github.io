package fr.unice.polytech.si5.rimel.mavenprofiles.categorize.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NoArgsConstructor;
import org.neo4j.ogm.annotation.*;

@RelationshipEntity(type = "PROPERTIES")
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class HasProperty {

	@Id
	@GeneratedValue
	private Long id;

	@StartNode
	private MavenProfile mavenProfile;

	@EndNode
	private MavenProperty mavenProperty;

	@Property
	private String name;

	@Property
	@Builder.Default
	private int weight = 1;

	@Property
	private String profileName;

	public MavenProfile getMavenProfile() {
		return mavenProfile;
	}

	public void setMavenProfile(MavenProfile mavenProfile) {
		this.mavenProfile = mavenProfile;
	}

	public MavenProperty getMavenProperty() {
		return mavenProperty;
	}

	public void setMavenProperty(MavenProperty mavenProperty) {
		this.mavenProperty = mavenProperty;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public int getWeight() {
		return weight;
	}

	public void setWeight(int weight) {
		this.weight = weight;
	}
}
